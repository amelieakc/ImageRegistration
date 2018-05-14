class CommandIterationUpdate : public itk::Command
{
 public:
  using Self = CommandIterationUpdate;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro( Self );
 protected:
  CommandIterationUpdate() {};
 public:
  using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
  using OptimizerPointer = const OptimizerType*;
  void Execute(itk::Object *caller, const itk::EventObject & event) override
  {
    Execute( (const itk::Object *)caller, event);
  }
  void Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    auto optimizer = static_cast< OptimizerPointer >( object );
    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
	return;
      }
    std::cout << optimizer->GetCurrentIteration() << " = ";
    std::cout << optimizer->GetValue() << " : ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;
  }
};
