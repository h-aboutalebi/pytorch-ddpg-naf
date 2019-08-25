

class Run_RL():

    def __init__(self,num_steps,update_interval,agent):
        self.num_steps=num_steps
        self.update_interval=update_interval
        self.agent=agent

    def run(self):
        for step_number in range(self.num_steps):
            
            if(step_number%self.update_interval==0):
                self.update()

