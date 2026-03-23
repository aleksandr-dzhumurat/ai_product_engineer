# Python objects model

### Dunder Methods

**Question:**

Consider the following Python class implementation:

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __len__(self):
        return 2

# Example usage
v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2

print(v1)
print(v2)
print(v3)
print(v1 == v2)
print(len(v1))

```

1. What will be the output of the code above?
2. Explain the purpose of each dunder method (`__add__`, `__repr__`, `__eq__`, and `__len__`) used in the `Vector` class.
3. What will happen if you try to add a `Vector` instance with an integer, e.g., `v1 + 5`? How can you handle such cases?
4. How does the `__eq__` method ensure correct comparison between `Vector` objects? What will happen if you compare a `Vector` instance with a non-`Vector` object?
5. What is the purpose of the `__repr__` method in this class, and how does it differ from the `__str__` method?

**Expected Answers:**

1. **Output:**
    
    The output of the code will be:
    
    ```
    Vector(1, 2)
    Vector(3, 4)
    Vector(4, 6)
    False
    2
    
    ```
    
    - `print(v1)` and `print(v2)` use `__repr__` to display the `Vector` instances.
    - `print(v3)` shows the result of adding `v1` and `v2` using the `__add__` method.
    - `print(v1 == v2)` evaluates to `False` because `v1` and `v2` have different `x` and `y` values.
    - `print(len(v1))` prints `2` because `__len__` returns `2`.
2. **Purpose of Each Dunder Method:**
    - **`__add__`**: Defines the behavior of the `+` operator for `Vector` objects. It adds corresponding `x` and `y` values of two `Vector` instances and returns a new `Vector`.
    - **`__repr__`**: Provides a string representation of the `Vector` object that is useful for debugging and logging. It returns a string that can be used to recreate the object.
    - **`__eq__`**: Defines equality comparison (`==`) between two `Vector` instances. It returns `True` if both `x` and `y` values are equal; otherwise, it returns `False`.
    - **`__len__`**: Returns the length of the `Vector` object, which is `2` in this case (the number of dimensions).
3. **Handling Addition with Non-`Vector` Objects:**
    
    Trying to add a `Vector` instance with a non-`Vector` object, such as an integer (`v1 + 5`), will return `NotImplemented`, which will lead to a `TypeError`. To handle such cases, you can add an additional check in the `__add__` method to handle or raise appropriate exceptions:
    
    ```python
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented
    
    ```
    
    You could also customize this behavior to raise a `TypeError` if desired.
    
4. **Comparison with Non-`Vector` Objects:**
    
    The `__eq__` method ensures correct comparison by checking if the other object is an instance of `Vector` and comparing `x` and `y` values. If you compare a `Vector` instance with a non-`Vector` object, it will return `NotImplemented`, which typically results in a `TypeError` if no alternative comparison is provided.
    
5. **Purpose of `__repr__` vs. `__str__`:**
    - **`__repr__`**: Provides a detailed string representation of the object, primarily for developers and debugging. It is meant to be unambiguous and, if possible, should allow the recreation of the object.
    - **`__str__`**: Provides a user-friendly string representation of the object, meant for end-users. It is usually more readable and less detailed than `__repr__`.
    
    If `__str__` were defined in this class, `print(v1)` would use `__str__` instead of `__repr__`.
    

This question tests the candidate’s understanding of dunder methods and their role in customizing behavior for Python objects, handling comparisons, and implementing meaningful representations.

**Question:** Consider the following Python class implementation:

```python
class Matrix:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return '\\n'.join([' '.join(map(str, row)) for row in self.data])

    def __add__(self, other):
        if isinstance(other, Matrix) and len(self) == len(other):
            return Matrix([[self[i][j] + other[i][j] for j in range(len(self[i]))] for i in range(len(self))])
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Matrix) and len(self) == len(other):
            return all(self[i] == other[i] for i in range(len(self)))
        return False

# Example usage
matrix1 = Matrix([[1, 2], [3, 4]])
matrix2 = Matrix([[5, 6], [7, 8]])
matrix_sum = matrix1 + matrix2

print(matrix1)
print(matrix2)
print(matrix_sum)
print(matrix1 == matrix2)
print(len(matrix1))

```

1. What will be the output of the code above?
2. Explain the purpose of each dunder method (`__getitem__`, `__setitem__`, `__len__`, `__str__`, `__add__`, and `__eq__`) used in the `Matrix` class.
3. What will happen if you try to add a `Matrix` instance with a non-`Matrix` object, such as an integer, e.g., `matrix1 + 5`? How is this handled?
4. How does the `__eq__` method ensure correct comparison between `Matrix` objects? What will happen if you compare two matrices of different sizes or with non-equal elements?
5. What is the difference between `__str__` and `__repr__` methods in this class, and how would the implementation of `__repr__` differ if it were included?

**Expected Answers:**

1. **Output:**
    
    The output of the code will be:
    
    ```
    1 2
    3 4
    5 6
    7 8
    6 8
    10 12
    False
    2
    ```
    
    - `print(matrix1)` and `print(matrix2)` use the `__str__` method to display the matrices.
    - `print(matrix_sum)` shows the result of adding `matrix1` and `matrix2` using the `__add__` method.
    - `print(matrix1 == matrix2)` evaluates to `False` because the matrices have different elements.
    - `print(len(matrix1))` prints `2`, the number of rows in `matrix1`.
2. **Purpose of Each Dunder Method:**
    - **`__getitem__`**: Allows indexing into the `Matrix` instance using square brackets (`matrix[i]`). Returns the row at index `i`.
    - **`__setitem__`**: Allows setting a value in the `Matrix` instance using square brackets (`matrix[i] = value`). Updates the row at index `i` with `value`.
    - **`__len__`**: Returns the number of rows in the `Matrix` instance. This is used to check the matrix dimensions.
    - **`__str__`**: Provides a human-readable string representation of the `Matrix`, formatting each row of the matrix as a string.
    - **`__add__`**: Defines the behavior of the `+` operator for `Matrix` objects. Adds corresponding elements of two matrices and returns a new `Matrix`.
    - **`__eq__`**: Defines equality comparison (`==`) between `Matrix` instances. Returns `True` if both matrices have the same dimensions and equal corresponding elements; otherwise, `False`.
3. **Handling Addition with Non-`Matrix` Objects:**
    
    Adding a `Matrix` instance with a non-`Matrix` object (e.g., `matrix1 + 5`) will return `NotImplemented`, which typically results in a `TypeError` if no alternative addition is defined. This is handled by returning `NotImplemented` in the `__add__` method:
    
    ```python
    def __add__(self, other):
        if isinstance(other, Matrix) and len(self) == len(other):
            return Matrix([[self[i][j] + other[i][j] for j in range(len(self[i]))] for i in range(len(self))])
        return NotImplemented
    
    ```
    
4. **Comparison with Non-Equal or Different Size Matrices:**
    
    The `__eq__` method compares matrices by checking their dimensions and then their contents. If two matrices have different sizes or different elements, the method will return `False`. If the matrices are of different sizes, the method will return `False` without comparing the elements.
    
5. **Difference Between `__str__` and `__repr__`:**
    - **`__str__`**: Provides a user-friendly string representation of the object, suitable for display. It is intended to be readable and understandable.
    - **`__repr__`**: Provides a detailed string representation of the object, often including more information useful for debugging. It should be unambiguous and, if possible, allow the object to be recreated.
    
    If `__repr__` were included, it might look like:
    
    ```python
    def __repr__(self):
        return f"Matrix({self.data})"
    
    ```
    
    This provides a more technical view of the matrix’s internal data structure.
    

This question tests the candidate’s understanding of various dunder methods and their roles in customizing object behavior, managing comparisons, and providing meaningful representations.

## Object-Oriented Programming

- **Encapsulation**: Hiding internal details and exposing only necessary parts (Access via method, not directly)
- **Abstraction**: Hiding complexity by providing a simple interface (Only care about the choose right method, not implementation)
- **Inheritance**: Allowing classes to inherit common behavior from other classes.
- **Polymorphism**: Allowing methods to have the same name but behave differently depending on the object.
- **Composition**: Building complex objects from simpler ones, promoting modular design.

### **Polymorphism and Inheritance**

**Question:**

Consider the following Python code that involves multiple classes and inheritance. Analyze the code and answer the related questions:

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalShelter:
    def __init__(self):
        self.animals = []

    def add_animal(self, animal):
        if not isinstance(animal, Animal):
            raise TypeError("Only objects of type Animal can be added")
        self.animals.append(animal)

    def make_all_speak(self):
        for animal in self.animals:
            print(f"{animal.name} says {animal.speak()}")

def main():
    shelter = AnimalShelter()
    dog = Dog(name="Buddy")
    cat = Cat(name="Whiskers")

    shelter.add_animal(dog)
    shelter.add_animal(cat)

    shelter.make_all_speak()

if __name__ == "__main__":
    main()

```

1. Explain how the principle of polymorphism is demonstrated in the `Animal`, `Dog`, and `Cat` classes. How does the `AnimalShelter` class utilize polymorphism?
2. What would happen if you tried to add an object of a class that is not derived from `Animal` to the `AnimalShelter`? Why does this behavior occur?
3. What is the purpose of the `raise NotImplementedError` statement in the `Animal` class? How does this design decision enforce the intended behavior of subclasses?
4. Discuss the role of the `isinstance` check in the `add_animal` method of the `AnimalShelter` class. Why is this check important and what are its benefits?
5. If you needed to extend this system to support other types of animals (e.g., birds or rabbits), how would you modify the existing code to accommodate these new types? Provide a code example demonstrating your changes.

**Expected Answers:**

1. **Polymorphism in `Animal`, `Dog`, and `Cat`:**
    - Polymorphism is demonstrated by the `speak` method in the `Animal` class, which is implemented differently in its subclasses (`Dog` and `Cat`). Each subclass provides its own version of the `speak` method.
    - The `AnimalShelter` class uses polymorphism by calling the `speak` method on `Animal` objects without needing to know the exact type of the animal. This allows the `make_all_speak` method to work with any subclass of `Animal` seamlessly.
2. **Adding Non-`Animal` Objects:**
    - If an object of a class not derived from `Animal` is added to the `AnimalShelter`, the `TypeError` will be raised due to the `isinstance` check in the `add_animal` method. This ensures that only valid `Animal` objects are added to the shelter.
    - This behavior occurs because `isinstance` verifies whether the object is an instance of the `Animal` class or its subclasses, ensuring type safety and correctness.
3. **Purpose of `raise NotImplementedError`:**
    - The `raise NotImplementedError` statement in the `Animal` class enforces that any subclass must implement the `speak` method. This design ensures that the method is not accidentally left unimplemented in subclasses, which would otherwise lead to runtime errors.
    - It acts as a placeholder, signaling that subclasses must provide their specific implementation of the method.
4. **Role of `isinstance` Check:**
    - The `isinstance` check in `add_animal` ensures that only instances of `Animal` or its subclasses are added to the shelter. This prevents incorrect types from being stored in the `animals` list, maintaining type consistency.
    - It provides safety by ensuring that operations performed on `animals` are valid for `Animal` objects, reducing potential errors and improving code reliability.
5. **Extending the System:**
    - To accommodate new types of animals, you would need to create additional subclasses of `Animal` (e.g., `Bird`, `Rabbit`) and implement their specific versions of the `speak` method.
    - Example code to add a `Bird` class:
        
        ```python
        class Bird(Animal):
            def speak(self):
                return "Tweet!"
        
        # Extend main() to include Bird
        def main():
            shelter = AnimalShelter()
            dog = Dog(name="Buddy")
            cat = Cat(name="Whiskers")
            bird = Bird(name="Tweety")
        
            shelter.add_animal(dog)
            shelter.add_animal(cat)
            shelter.add_animal(bird)
        
            shelter.make_all_speak()
        
        if __name__ == "__main__":
            main()
        
        ```
        

This question assesses the candidate’s understanding of OOP principles, including polymorphism, type safety, and the design of class hierarchies. It also evaluates their ability to extend and adapt existing code.

### Encapsulation

**Question:**

Consider the following Python code that demonstrates the use of encapsulation:

```python
class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
        else:
            print("Deposit amount must be positive")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
        else:
            print("Invalid withdrawal amount")

    def get_balance(self):
        return self.__balance

# Testing the BankAccount class
account = BankAccount(account_number="12345678")
account.deposit(1000)
account.withdraw(500)
print(account.get_balance())  # Output: 500

```

1. Explain how encapsulation is demonstrated in the `BankAccount` class. What are the benefits of using encapsulation in this context?
2. What would happen if you tried to access the `__balance` attribute directly from outside the `BankAccount` class? Provide an example to illustrate this.
3. Why is the `__balance` attribute marked as private with double underscores? How does this impact the accessibility of the attribute?
4. How would you modify the `BankAccount` class to allow for interest to be added to the account balance, while still adhering to the principles of encapsulation? Provide a code example.

**Expected Answers:**

1. **Encapsulation in `BankAccount`:**
    - Encapsulation is demonstrated by making the `__balance` attribute private (with double underscores), meaning it cannot be accessed directly from outside the `BankAccount` class. This ensures that the balance is only modified through controlled methods (`deposit` and `withdraw`), which include validation checks.
    - Benefits of encapsulation include protecting the internal state of an object, controlling access to data, and reducing the risk of unintended interference or misuse of the data.
2. **Accessing `__balance` Directly:**
    - If you try to access the `__balance` attribute directly from outside the `BankAccount` class, you will encounter an `AttributeError` because the attribute is private. For example:
        
        ```python
        print(account.__balance)  # This will raise an AttributeError
        
        ```
        
    - This error occurs because Python performs name mangling for private attributes by adding a prefix (e.g., `_BankAccount__balance`), making it difficult to access the attribute directly.
3. **Private Attribute with Double Underscores:**
    - The `__balance` attribute is marked private with double underscores to signal that it should not be accessed directly from outside the class. This triggers name mangling, where Python changes the attribute name to include the class name as a prefix.
    - This impacts accessibility by preventing accidental or unauthorized modifications from outside the class, ensuring that the attribute is only accessed and modified through class-defined methods.
4. **Adding Interest while Maintaining Encapsulation:**
    - To add interest to the account balance while adhering to encapsulation principles, you can provide a method in the `BankAccount` class that calculates and adds interest. This maintains control over how the balance is modified. Example code:
        
        ```python
        class BankAccount:
            def __init__(self, account_number, balance=0):
                self.account_number = account_number
                self.__balance = balance  # Private attribute
        
            def deposit(self, amount):
                if amount > 0:
                    self.__balance += amount
                else:
                    print("Deposit amount must be positive")
        
            def withdraw(self, amount):
                if 0 < amount <= self.__balance:
                    self.__balance -= amount
                else:
                    print("Invalid withdrawal amount")
        
            def get_balance(self):
                return self.__balance
        
            def add_interest(self, rate):
                if rate > 0:
                    interest = self.__balance * rate / 100
                    self.__balance += interest
                else:
                    print("Interest rate must be positive")
        
        # Testing the updated BankAccount class
        account = BankAccount(account_number="12345678")
        account.deposit(1000)
        account.add_interest(5)  # Adding 5% interest
        print(account.get_balance())  # Output: 1050
        
        ```
        

This question tests the candidate's understanding of encapsulation, attribute privacy, and controlled access, as well as their ability to extend functionality while adhering to encapsulation principles.

### Inheritance

Consider the following Python code that demonstrates inheritance:

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "Some generic animal sound"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed

    def speak(self):
        return "Woof!"

class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name)
        self.color = color

    def speak(self):
        return "Meow!"

# Create instances of Dog and Cat
dog = Dog(name="Rex", breed="Labrador")
cat = Cat(name="Whiskers", color="Gray")

# Demonstrate polymorphism
animals = [dog, cat]
for animal in animals:
    print(f"{animal.name} says: {animal.speak()}")

```

1. Explain how inheritance is demonstrated in the provided code. What is the role of the `super()` function in the `Dog` and `Cat` classes?
2. What is the output of the provided code when executed? Explain why the output is as it is.
3. How does polymorphism work in this code snippet? Provide an example of how polymorphism is demonstrated in the code.
4. What changes would you make to the `Animal` class if you wanted to add a new method that should be implemented by all subclasses? Illustrate your changes with code.

**Expected Answers:**

1. **Inheritance Explanation:**
    - Inheritance is demonstrated by the `Dog` and `Cat` classes inheriting from the `Animal` class. This allows `Dog` and `Cat` to reuse and extend the functionality of `Animal`.
    - The `super()` function is used to call the `__init__` method of the `Animal` class from within the `__init__` methods of `Dog` and `Cat`. This ensures that the `name` attribute is properly initialized for instances of `Dog` and `Cat`.
2. **Output Explanation:**
    - The output of the code is:
        
        ```
        Rex says: Woof!
        Whiskers says: Meow!
        
        ```
        
    - The output is as such because the `speak()` method is overridden in both `Dog` and `Cat` classes. The `speak()` method of `Dog` returns "Woof!" and the `speak()` method of `Cat` returns "Meow!". When the `speak()` method is called on `Dog` and `Cat` instances, the overridden methods are used due to polymorphism.
3. **Polymorphism Explanation:**
    - Polymorphism is demonstrated by treating objects of different classes (`Dog` and `Cat`) through a common interface (`Animal`). Despite being different classes, they can be used interchangeably through their common `speak()` method.
    - Example of polymorphism: The loop iterates over a list of `Animal` objects (`dog` and `cat`) and calls `speak()` on each. Each object's version of `speak()` is invoked, demonstrating polymorphism.
4. **Adding a Method to `Animal`:**
    
    To add a new method that should be implemented by all subclasses, you can define an abstract method in the `Animal` class using the `abc` module. Example code:
    
    ```python
    from abc import ABC, abstractmethod
    
    class Animal(ABC):
        def __init__(self, name):
            self.name = name
    
        @abstractmethod
        def speak(self):
            pass
    
        @abstractmethod
        def eat(self):
            """Method that should be implemented by all subclasses"""
            pass
    
    class Dog(Animal):
        def __init__(self, name, breed):
            super().__init__(name)
            self.breed = breed
    
        def speak(self):
            return "Woof!"
    
        def eat(self):
            return f"{self.name} is eating dog food."
    
    class Cat(Animal):
        def __init__(self, name, color):
            super().__init__(name)
            self.color = color
    
        def speak(self):
            return "Meow!"
    
        def eat(self):
            return f"{self.name} is eating cat food."
    
    # Create instances of Dog and Cat
    dog = Dog(name="Rex", breed="Labrador")
    cat = Cat(name="Whiskers", color="Gray")
    
    # Demonstrate the new method
    animals = [dog, cat]
    for animal in animals:
        print(f"{animal.name} says: {animal.speak()}")
        print(animal.eat())
    
    ```
    
    In this updated code, the `Animal` class now includes an abstract method `eat()`, which all subclasses must implement. This ensures that each subclass provides its own implementation of the `eat()` method.