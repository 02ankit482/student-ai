class history_manager:
    def __init__(self):#constructor
        self.history={}
    def add_message(self, user_id, message):# adds meesgae by user id
        if user_id not in self.history:
            self.history[user_id]=[]
        self.history[user_id].append(message)
    def get_histort(self, user_id):
        return self.history.get(user_id,[]) 
    # history done
       