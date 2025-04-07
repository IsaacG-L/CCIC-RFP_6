from gymnasium import spaces

class ActionSpaces:
    def __init__(self, action_spaces : list[spaces.Space] = None):
        self.action_spaces = action_spaces

    def AddSpace(self, space : spaces.Space):
        self.action_spaces.append(space)

    def SetSpace(self, i : int, space : spaces.Space):
        self.action_spaces[i] = space

    def GetSpace(self, i):
        return self.action_spaces[i]
    
    def Sample(self, i):
        return self.action_spaces[i].sample()

    def SampleAll(self):
        results = [None] * len(self.action_spaces)
        for i in range(len(self.action_spaces)):
            results[i] = self.action_spaces[i].sample()

        return results