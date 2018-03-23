

class FixedProfile():

    def choose_action(self, t):

        if t <= 66:
            return 0/90
        elif t <= 2 * 66:
            return 70/90
        elif t <= 3 * 66:
            return 60/90
        elif t <= 4 * 66:
            return 50/90
        elif t< 327:
            return 45/90
        else:
            return 1

