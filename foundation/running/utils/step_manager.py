class StepManager:
    def __init__(self, current_step: int = 0, start_processing_step: int = 0):
        self.global_step = current_step
        self.start_processing_step = start_processing_step
        self.__judge_step = self.__judge_step_before_processing
    
    @property
    def enable_processing(self):
        return self.global_step > self.start_processing_step
    
    def reset(self, current_step: int | None = None, start_processing_step: int | None = None):
        if current_step is not None:
            self.global_step = current_step
        if start_processing_step is not None:
            self.start_processing_step = start_processing_step
        self.__judge_step = self.__judge_step_before_processing
    
    def __judge_step(self):
        pass
    
    def __judge_step_before_processing(self):
        if self.enable_processing:
            self.__judge_step = self.__judge_step_normal
    
    def __judge_step_normal(self):
        return

    def __enter__(self):
        self.__judge_step()
        return self

    def step(self):
        self.global_step += 1
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False