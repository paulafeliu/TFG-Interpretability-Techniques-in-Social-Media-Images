import torch.nn as nn

class SingleOutputWrapper(nn.Module):
    def __init__(self, model, output_index=0):
        """
        Wraps a multi-output model so that only the output with the 
        specified index is returned.
        
        Parameters:
            model: The original multi-task model.
            output_index: The index of the task output you want to explain.
        """
        super(SingleOutputWrapper, self).__init__()
        self.model = model
        self.output_index = output_index

    def forward(self, x):
        outputs = self.model(x)
        # Return only the selected output.
        return outputs[self.output_index]
