class Game:

    def __init__(self):
        self.playerChannel = Interaction()
        self.qn1 = Questionnaire1()
        self.qn2 = Questionnaire2()
        self.qn3 = Questionnaire3()
        self.qn4 = Questionnaire4()
        self.color = Color()

    def play(self):
        self.playerChannel.expressWelcome()
        self.playerInfo = Player()
        di1 = self.qn1.five_question()
        di2 = self.qn2.five_question()
        di3 = self.qn3.five_question()
        di4 = self.qn4.five_question()
        self.playerInfo.set_character(di1, di2, di3, di4)
        self.playerInfo.printplayer()
        self.color.print_color_info(di1, di2, di3, di4)


class Interaction:

    def __init__(self):
        self.messages = "Welcome to our personality test"

    def expressWelcome(self):
        print(self.messages)


class Player:
    def __init__(self):
        self.name = input("What is your name?")

    def set_character(self, di1, di2, di3, di4):
        self.di1 = di1
        self.di2 = di2
        self.di3 = di3
        self.di4 = di4
        return self

    def printplayer(self):
        print(
            "Thank you for taking the test. The personalities of yours are " + self.di1 + ", " + self.di2 + ", " + self.di3 + " and " + self.di4)


class Questionnaire1:
    def __init__(self):
        self.total_score = 0
        self.question_list = [
            "I prefer one-on-one conversations to group activities.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I often prefer to express myself in writing.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I enjoy solitude.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I seem to care about wealth, fame, and status less than my peers.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I dislike small talk, but I enjoy talking in-depth about topics that matter to me.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree"
        ]

    def five_question(self):
        nature = ''
        for q in self.question_list:
            print(q)
            while True:
                try:
                    option = int(input("I would choose no."))
                    break
                except:
                    print("please enter number 1 to 5")
            while option > 5 or option < 1:
                option = int(input("please enter number 1 to 5 - I would choose no."))
            self.total_score += option
        if self.total_score <= 0:
            print("you are confident")
            nature = "confident"
        elif self.total_score > 0:
            print("you are shy")
            nature = "shy"
        return nature


class Questionnaire2:
    def __init__(self):
        self.total_score = 0
        self.question_list = [
            "I like to plan ahead.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I am always on time for meetings and dates.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree ",

            "I like to be aware of what the menu is going to be and decide my order before-hand if we are going out.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree ",

            "I prefer well-structured day/routine.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5.strongly disagree ",

            "I never lose my personal items.\n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5.strongly disagree "
        ]

    def five_question(self):
        nature = ''
        for q in self.question_list:
            print(q)
            while True:
                try:
                    option = int(input("I would choose no."))
                    break
                except:
                    print("please enter number 1 to 5")
            while option > 5 or option < 1:
                option = int(input("please enter number 1 to 5 - I would choose no."))
            self.total_score += option
        if self.total_score <= 0:
            print("you are spontaneous")
            nature = "spontaneous"
        elif self.total_score > 0:
            print("you are organized")
            nature = "organized"
        return nature


class Questionnaire3:
    def __init__(self):
        self.total_score = 0
        self.question_list = [
            "I like to take a lot of pictures when I travel.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I am a imaginative person.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I like to look up for things I am curious about.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I don't like to be in new environments.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I like to learn new languages.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree"
        ]

    def five_question(self):
        nature = ''
        for q in self.question_list:
            print(q)
            while True:
                try:
                    option = int(input("I would choose no."))
                    break
                except:
                    print("please enter number 1 to 5")
            while option > 5 or option < 1:
                option = int(input("please enter number 1 to 5 - I would choose no."))
            self.total_score += option
        if self.total_score <= 0:
            print("you are curious")
            nature = "curious"
        elif self.total_score > 0:
            print("you are indifferent")
            nature = "indifferent"
        return nature


class Questionnaire4:
    def __init__(self):
        self.total_score = 0
        self.question_list = [
            "I prefer the group projects in our school.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "When I go to make a decision, I am thinking of how it will impact others rather than me.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I always form my opinions based on others.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I don't like to stand out in a crowd.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree",

            "I am accustomed to long-term relationships.\
            \n 1. strongly agree \n 2. agree \n 3. neutral \n 4. disagree \n 5. strongly disagree"
        ]

    def five_question(self):
        nature = ''
        for q in self.question_list:
            print(q)
            while True:
                try:
                    option = int(input("I would choose no."))
                    break
                except:
                    print("please enter number 1 to 5")
            while option > 5 or option < 1:
                option = int(input("please enter number 1 to 5 - I would choose no."))
            self.total_score += option
        if self.total_score <= 0:
            print("you are individualistic")
            nature = "individualistic"
        elif self.total_score > 0:
            print("you are collective centric")
            nature = "collective centric"
        return nature


class Color():
    def __init__(self):
        Maroon = ("confident", "organized", "curious", "collective centric")
        Orange = ("confident", "organized", "curious", "individualistic")
        Red = ("confident", "organized", "indifferent", "collective centric")
        Gold = ("confident", "organized", "indifferent", "individualistic")
        Yellow = ("confident", "spontaneous", "curious", "collective centric")
        Olive = ("confident", "spontaneous", "curious", "individualistic")
        Green = ("confident", "spontaneous", "indifferent", "individualistic")
        Turquoise = ("confident", "spontaneous", "indifferent", "collective centric")
        Lavender = ("shy", "organized", "indifferent", "individualistic")
        Blue = ("shy", "organized", "curious", "collective centric")
        Purple = ("shy", "organized", "curious", "collective centric")
        Pink = ("shy", "organized", "indifferent", "collective centric")
        White = ("shy", "spontaneous", "curious", "collective centric")
        Black = ("shy", "spontaneous", "curious", "individualistic")
        Gray = ("shy", "spontaneous", "indifferent", "individualistic")
        Brown = ("shy", "spontaneous", "indifferent", "collective centric")

        listofcolors = [Maroon, Red, Orange, Gold, Yellow, Olive, Green, Turquoise, Lavender, Blue, Purple, Pink, White,
                        Black, Gray, Brown]

        col_names = ["Maroon", "Red", "Orange", "Gold", "Yellow", "Olive", "Green", "Turquoise", "Lavender", "Blue",
                     "Purple", "Pink", "White", "Black", "Gray", "Brown"]

        col_description = [
            "You are an open person that likes to take the initiative to start conversations and make new friends around the world. Although you like to plan ahead and prefer a structured day you don't shy away from trying out new things and involving your friends in all your creative ideas that you plan.",
            "You are a very social and outgoing person. The warmth you radiate makes people want to be around you. You don't like being late and always try to structure your day in order to achieve maximum results. Even though you are very sociable you still like to take time off the day to spend some time alone and think about what's best for you.",
            "You are a strong willed person that screams energy and passion. You don’t like it if things don’t go your way and don’t have a problem being the center of attention. The precise routine that you have in your daily life gives you a sense of organization and power.",
            "You are a very individualistic person that stands a confident and firm group, no matter what situation gets thrown at you. Having things your way makes you feel empowered and organized. You don’t like the idea of someone else putting their views and opinions on you and would rather work alone than.",
            "You are an enthusiastic person that has a talent to cheer people up and have lots of fun. Your kind and curious nature inspires you to take life on a light note. You’re always up for the craziest and most spontaneous adventures your friends suggest and can’t resist capturing your life in photographs and posts on social media.",
            "You’re a sophisticated person that can work very well on their own but also loves to spend time with family and friends. Your drive for adventure makes you a very curious and open person, although you sometimes rather explore the world with soly the peace of your own mind.",
            "You are a very encouraging and calm person that loves to spend quality time alone and with friends and family. Your stable and confident personality makes you a reliable friend that loves to explore new things. ",
            "You are a very empathic and caring person that doesn't shy away from standing up for the people that matter to you most. You’re a very ambitious person that loves to plan and go on trips and holidays with your friends.",
            "Your calm and collected nature draws people to you for security and comfort. You love to be surrounded by your close circle of friends and family but can also stay alone and find various ways of spending your day being productive.",
            "You are a very critical thinker and analyze everything from a safe distance. Your ambition to live an organized and balanced life drives you towards likeminded people that share your curious thinking.",
            "You have a peaceful and sensitive character that makes people trust you and reach out to you for support. You are a collective person that enjoys being around people and that loves to have meaningful conversations.",
            "You are a loving and sweet person that loves to talk. Always being on top of things you are very organized and respect other people's time and values. Your indifferent nature makes you focus on your own goals and projects.",
            "Your pure and caring personality traits make people feel secure around you. You are a trustable person that people love to hangout with for spontaneous adventures and trips around the world.",
            "You radiate professionalism and power while staying your individualistic and collective self. Your sophisticated personality makes people trust deeply in your opinion and the choices you make.",
            "You are a well balanced and practical person that loves to have their own time to spend on new ideas. Your neutral and fair personality makes you a good mediator during other peoples quarrels.",
            "You give a trustful and compassionate image that draws people to you in times of comfort. While loving to go on spontaneous adventures with your close friends and family you also don't shy away from spending time in big groups of people and to make new friends."
        ]

        dic = {}
        for i in range(len(listofcolors)):
            dic[listofcolors[i]] = "You are of the color " + col_names[i] + ". " + col_description[i]
        self.color_dic = dic

    def print_color_info(self, di1, di2, di3, di4):
        print(self.color_dic[(di1, di2, di3, di4)])


if __name__ == '__main__':
    game = Game()
    game.play()
