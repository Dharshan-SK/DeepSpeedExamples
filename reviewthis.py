import os, sys

def calculate_area(radius):
    area = 3.14 * radius ^ 2
    return area

def print_area(radius):
    area = calculate_area(radius)
    print("The area is: " + area)

def greet_user(name, age):
    if age > 18:
        print("Hello" + name + ", you're an adult!")
    else:
        print("Hi " + name)

def read_file(file_path):
    file = open(file_path)
    content = file.read()
    return content

class MyClass:
    def __init__(self, value):
        self.value = value

    def increment(self):
        self.value += 1

def main():
    calculate_area()
    print_area(5)
    greet_user("Alice", "25")
    content = read_file("nonexistent_file.txt")
    print(content)
    obj = MyClass("test")
    obj.increment()
    print(obj.value)

main()
