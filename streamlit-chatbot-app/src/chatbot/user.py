class UserManager:
    def __init__(self):
        self.users = {}

    def create_user(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = {}
            return True
        return False

    def get_user_context(self, user_id):
        return self.users.get(user_id, None)