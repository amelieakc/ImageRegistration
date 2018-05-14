#include "itkImageRegistrationMethodv4.h"
#include "itkTranslationTransform.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "ImageRegistrationMethod.h"

int main( int argc, char *argv[] )
{

  const unsigned int Dimension = 2;
  using PixelType = float;

  // create fixed and moving image types
  using FixedImageType = itk::Image<PixelType, Dimension>;
  using MovingImageType = itk::Image<PixelType, Dimension>;

  // transformer 
  using TransformType = itk::TranslationTransform<double, Dimension>;

  // optimizer using gradient descent 
  using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;

  using MetricType = itk::MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType>;

  using RegistrationType = itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType, TransformType>;

  MetricType::Pointer metric = MetricType::New();
  OptimizerType::Pointer optimizer = OptimizerType::New();
  RegistrationType::Pointer registration = RegistrationType::New();

  // attach metric and optimizer to the registration object
  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);

  using FixedLinearInterpolatorType = itk::LinearInterpolateImageFunction<FixedImageType,double>;
  using MovingLinearInterpolatorType = itk::LinearInterpolateImageFunction<MovingImageType, double>;

  FixedLinearInterpolatorType::Pointer fixedInterpolator = FixedLinearInterpolatorType::New();
  MovingLinearInterpolatorType::Pointer movingInterpolator = MovingLinearInterpolatorType::New();

  // Interpolator allows use to compare similarity using metric
  metric->SetFixedInterpolator(fixedInterpolator);
  metric->SetMovingInterpolator(movingInterpolator);

  // read in fixed and moving images
  using FixedImageReaderType = itk::ImageFileReader<FixedImageType>;
  using MovingImageReaderType = itk::ImageFileReader<MovingImageType>;
  FixedImageReaderType::Pointer fixedImageReader = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  fixedImageReader->SetFileName(argv[1]);
  movingImageReader->SetFileName(argv[2]);

  // set fixed and moving image to registration object
  registration->SetFixedImage(fixedImageReader->GetOutput());
  registration->SetMovingImage(movingImageReader->GetOutput());

  TransformType::Pointer movingInitialTransform = TransformType::New();
  TransformType::ParametersType initialParameters(movingInitialTransform->GetNumberOfParameters());

  // Set initialParameters to 0.0
  initialParameters[0] = 0.0;
  initialParameters[1] = 0.0; 
  movingInitialTransform->SetParameters(initialParameters);
  registration->SetMovingInitialTransform(movingInitialTransform);

  TransformType::Pointer identityTransform = TransformType::New();
  identityTransform->SetIdentity();
  registration->SetFixedInitialTransform(identityTransform);

  // set learning rate for gradient descent
  optimizer->SetNumberOfIterations(300);
  optimizer->SetLearningRate(3);
  optimizer->SetMinimumStepLength(0.01);
  optimizer->SetRelaxationFactor(0.3);
 
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver(itk::IterationEvent(), observer);

  const unsigned int numberOfLevels = 1;
  RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize(1);
  shrinkFactorsPerLevel[0] = 1;

  RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize(1);
  smoothingSigmasPerLevel[0] = 0;

  registration->SetNumberOfLevels(numberOfLevels);
  registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
  registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

  // run the update to perform optimization
  registration->Update();

  // Using the resulting transformer, convert image which will generate the output image
  TransformType::ConstPointer transform = registration->GetTransform();

  TransformType::ParametersType finalParameters = transform->GetParameters();
  const double TranslationAlongX = finalParameters[0];
  const double TranslationAlongY = finalParameters[1];

  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
 
  using CompositeTransformType = itk::CompositeTransform<double, Dimension>;
  CompositeTransformType::Pointer outputCompositeTransform = CompositeTransformType::New();
  outputCompositeTransform->AddTransform(movingInitialTransform);
  outputCompositeTransform->AddTransform(registration->GetModifiableTransform());

  using ResampleFilterType = itk::ResampleImageFilter<MovingImageType, FixedImageType>;

  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetInput(movingImageReader->GetOutput());

  resampler->SetTransform(outputCompositeTransform);

  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
  resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
  resampler->SetOutputOrigin(fixedImage->GetOrigin());
  resampler->SetOutputSpacing(fixedImage->GetSpacing());
  resampler->SetOutputDirection(fixedImage->GetDirection());
  resampler->SetDefaultPixelValue(100);

  using OutputPixelType = unsigned char;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using CastFilterType = itk::CastImageFilter<FixedImageType, OutputImageType>;
  using WriterType = itk::ImageFileWriter<OutputImageType>;

  WriterType::Pointer writer = WriterType::New();
  CastFilterType::Pointer caster = CastFilterType::New();

  writer->SetFileName(argv[3]);

  caster->SetInput(resampler->GetOutput());
  writer->SetInput(caster->GetOutput());
  writer->Update();

  return 0;
}