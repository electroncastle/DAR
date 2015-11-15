
class DataItem:

    def __init__(self, name, color):
        self.name = name
        self.color = color

    def set(self, name, color):
        self.name = name
        self.color = color

    # This overrides default "to string" function
    # This is for example used if you do
    # print myClassInstance
    def __str__(self):
        return "[DataItem] name="+self.name+"  color="+str(self.color)




def getData():

    # Create some random data
    data = []
    for i in range(10):
        data.append(DataItem( "Name "+str(i), [i,i,i] ))

    return data


# Entry point
if __name__ == "__main__":

    data = getData()
    for item in data:
        # print automatically sends '\n'
        # If you don't want that put comma at the end e.g.
        # print item,
        print item
