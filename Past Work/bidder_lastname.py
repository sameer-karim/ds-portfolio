import random

class Bidder:
    def __init__(self, num_users, num_rounds):
        self.num_users = num_users
        self.num_rounds = num_rounds
    
    def bid(self, user_id):
        # bid a random amount between 0 and 1000 dollars
        return round(random.uniform(0, 1000), 3)
    
    def notify(self, auction_winner, price, clicked):
        if auction_winner:
            print(f'The winning bid is {price}')
            if clicked is not None:  # Check if a click value is provided
                return 1 if clicked else 0
            return 0
        return 0

    def __str__(self):
        return f"Bidder with users: {self.num_users}"

    def __repr__(self):
        return f"Bidder({self.num_users})"

