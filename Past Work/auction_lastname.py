import random

class User:
    def __init__(self):
        '''Assigns a hidden probability to the user'''
        self.__probability = random.uniform(0, 1)
    
    # Simulates if a user clicks on the ad based on established probability 
    def show_ad(self):
        '''If the probability is greater than or equal to the show ad function, the user is 'shown' the ad'''
        return random.random() <= self.__probability

    def __str__(self):
        return f"User with click probability: {self.__probability:.3f}"

    def __repr__(self):
        return f"User({self.__probability:.3f})"

class Auction:
    def __init__(self, users, bidders):
        '''
        Args: self, users, bidders
        Returns: defines users and bidders, as well as keeps track of the bidder's balances
        '''
        self.users = users
        self.bidders = bidders
        self.balances = {bidder: 0 for bidder in bidders}
    
    def execute_round(self):
        '''
        Executes one round of the auction and updates balances for each bidder

        Args: self

        Returns: A winner and a winning price 
        '''
        for bidder in self.bidders:  # Iterate over all bidders
            num_users = len(self.users)  # Get the number of users
            user_id = random.randint(0, num_users - 1)  # Select a random user for each bidder
            user = self.users[user_id]  # Assign user IDs to each user

            # Establishes a dictionary with bidders and their balances
            bids = {bidder: bidder.bid(user_id) for bidder in self.bidders}
            winner = max(bids, key=bids.get)
            winning_price = max(bids.values())
            clicked = user.show_ad() if winner.bid(user_id) else None

            winner.notify(True, winning_price, clicked)

            for other_bidder in self.bidders:
                if other_bidder != winner:
                    other_bidder.notify(False, winning_price, None)
            
        
    def update_balances(self,user_id):
        '''
        Updates balances after one round of bidding

        Args: self

        Returns: Changes balances according to the execute round method
        '''
        for bidder in self.bidders:
            bidder.balances[bidder] -= bidder.bid(user_id)

