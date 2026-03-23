# Python design patterns



### SOLID

The SOLID principles are a set of five design principles in object-oriented programming that aim to make software designs more understandable, flexible, and maintainable. The principles are particularly important in ensuring that code is easier to manage and extend over time.

SOLID Principles

1. **Single Responsibility Principle (SRP):**
    - **Definition:** A class should have only one reason to change, meaning it should have only one job or responsibility.
    - **Explanation:** Each class should be responsible for a single part of the functionality provided by the software. This makes the system easier to understand and modify, as changes in one part of the system won't affect other parts.
2. **Open/Closed Principle (OCP):**
    - **Definition:** Software entities (classes, modules, functions, etc.) should be open for extension but closed for modification.
    - **Explanation:** You should be able to add new functionality to a class without altering its existing behavior. This can be achieved through mechanisms like inheritance, interfaces, and abstract classes.
3. **Liskov Substitution Principle (LSP):**
    - **Definition:** Objects of a superclass should be replaceable with objects of a subclass without affecting the correctness of the program.
    - **Explanation:** If a class `B` is a subclass of class `A`, you should be able to use `B` wherever `A` is expected without introducing errors. This principle ensures that a subclass does not change the behavior expected from the superclass.
4. **Interface Segregation Principle (ISP):**
    - **Definition:** No client should be forced to depend on methods it does not use.
    - **Explanation:** Instead of having a single, large interface, it's better to have multiple smaller, more specific interfaces. This way, classes can implement only the methods they actually need, leading to more modular and decoupled systems.
5. **Dependency Inversion Principle (DIP):**
    - **Definition:** High-level modules should not depend on low-level modules. Both should depend on abstractions. Additionally, abstractions should not depend on details. Details should depend on abstractions.
    - **Explanation:** The idea is to reduce the coupling between different pieces of code by relying on interfaces or abstract classes, rather than concrete implementations. This makes the system more flexible and easier to change.

Benefits of SOLID Principles

- **Maintainability:** The code is easier to modify and extend without introducing bugs.
- **Reusability:** Smaller, more focused classes and interfaces can be reused in different contexts.
- **Testability:** Code that adheres to SOLID principles is generally easier to test, as dependencies are minimized and well-defined.
- **Understandability:** The system is easier to understand because classes and modules are designed with a clear purpose in mind.

Following the SOLID principles helps create systems that are easier to develop, maintain, and scale over time.

### Single responsibility

Here’s a Python class that violates the SRP by handling both user authentication and email notifications:

```python
class UserManager:
    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email

    def authenticate_user(self):
        # Check if the username and password match the records
        if self.username == "admin" and self.password == "admin123":
            return True
        return False

    def send_welcome_email(self):
        # Simulate sending an email
        if self.authenticate_user():
            print(f"Sending welcome email to {self.email}")
        else:
            print("Authentication failed. Cannot send email.")

```

Issues with the Code

This `UserManager` class violates the SRP because it has two responsibilities:

1. **User Authentication**: The `authenticate_user` method checks the username and password.
2. **Sending Email Notifications**: The `send_welcome_email` method sends an email if the user is authenticated.

If you need to change how emails are sent or how users are authenticated, you’d have to modify this class, which could introduce bugs or side effects.

Refactoring to Follow SRP

To adhere to the Single Responsibility Principle, split the responsibilities into separate classes:

```python
class UserAuthenticator:
    def __init__(self, username, password):
        self.username = username
        self.password = password

    def authenticate(self):
        # Check if the username and password match the records
        return self.username == "admin" and self.password == "admin123"

class EmailService:
    def send_welcome_email(self, email):
        # Simulate sending an email
        print(f"Sending welcome email to {email}")

# Now, the UserManager class is only responsible for managing user-related tasks
class UserManager:
    def __init__(self, username, password, email):
        self.authenticator = UserAuthenticator(username, password)
        self.email_service = EmailService()
        self.email = email

    def register_user(self):
        if self.authenticator.authenticate():
            self.email_service.send_welcome_email(self.email)
        else:
            print("Authentication failed. Cannot send email.")

```

Benefits of the Refactor

1. **Single Responsibility**: Each class now has a single responsibility:
    - `UserAuthenticator`: Handles user authentication.
    - `EmailService`: Handles sending emails.
    - `UserManager`: Coordinates the interaction between the authenticator and the email service.
2. **Easier to Maintain**: Changes in how users are authenticated or how emails are sent won’t affect the other functionality.
3. **Reusability**: The `UserAuthenticator` and `EmailService` classes can be reused in other parts of the system without modification.

### Open/Closed Principle

Example that Breaks the Open/Closed Principle (OCP)

Here’s a Python class that violates the OCP by requiring modification every time a new payment method is added:

```python
class PaymentProcessor:
    def process_payment(self, payment_method, amount):
        if payment_method == "credit_card":
            self.process_credit_card(amount)
        elif payment_method == "paypal":
            self.process_paypal(amount)
        else:
            raise ValueError("Unsupported payment method")

    def process_credit_card(self, amount):
        print(f"Processing credit card payment of ${amount}")

    def process_paypal(self, amount):
        print(f"Processing PayPal payment of ${amount}")

```

Issues with the Code

This `PaymentProcessor` class violates the OCP because it needs to be modified every time a new payment method is added. This is not scalable and increases the risk of introducing bugs when making changes.

Refactoring to Follow OCP

To adhere to the Open/Closed Principle, use polymorphism to allow new payment methods to be added without modifying the existing code:

```python
from abc import ABC, abstractmethod

# Abstract base class for payment methods
class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount):
        pass

# Concrete class for credit card payments
class CreditCardPayment(PaymentMethod):
    def process(self, amount):
        print(f"Processing credit card payment of ${amount}")

# Concrete class for PayPal payments
class PayPalPayment(PaymentMethod):
    def process(self, amount):
        print(f"Processing PayPal payment of ${amount}")

# PaymentProcessor class that is open for extension but closed for modification
class PaymentProcessor:
    def process_payment(self, payment_method: PaymentMethod, amount):
        payment_method.process(amount)

# Example usage
processor = PaymentProcessor()
processor.process_payment(CreditCardPayment(), 100)
processor.process_payment(PayPalPayment(), 200)

```

Benefits of the Refactor

1. **Open for Extension**: You can now add new payment methods (e.g., `BitcoinPayment`, `ApplePayPayment`) by simply creating new classes that implement the `PaymentMethod` interface, without modifying the `PaymentProcessor` class.
2. **Closed for Modification**: The `PaymentProcessor` class does not need to change when new payment methods are added. This minimizes the risk of bugs and makes the system more maintainable.
3. **Scalability**: The code is easier to scale as new features can be added by extending the system rather than changing its core functionality.

### Liskov Substitution Principle (LSP)

Example that Breaks the Liskov Substitution Principle (LSP)

Here’s a Python example that violates the Liskov Substitution Principle:

```python
class Bird:
    def fly(self):
        return "Flying"

class Duck(Bird):
    def quack(self):
        return "Quacking"

class Ostrich(Bird):
    def fly(self):
        raise NotImplementedError("Ostriches can't fly")

def make_bird_fly(bird: Bird):
    return bird.fly()

# Usage
duck = Duck()
ostrich = Ostrich()

print(make_bird_fly(duck))  # Output: Flying
print(make_bird_fly(ostrich))  # Raises NotImplementedError

```

Issues with the Code

- The `Ostrich` class violates the LSP because it's a subclass of `Bird`, but it does not correctly implement the behavior expected from the `Bird` class (i.e., the ability to fly).
- The `make_bird_fly` function expects any `Bird` object to be able to fly, but passing an `Ostrich` object results in an error, breaking the program's correctness.

Refactoring to Follow LSP

To adhere to the Liskov Substitution Principle, you can refactor the code so that subclasses correctly implement the behavior expected by the superclass:

```python
from abc import ABC, abstractmethod

class Bird(ABC):
    @abstractmethod
    def move(self):
        pass

class FlyingBird(Bird):
    def move(self):
        return "Flying"

class Duck(FlyingBird):
    def quack(self):
        return "Quacking"

class Ostrich(Bird):
    def move(self):
        return "Running"

def make_bird_move(bird: Bird):
    return bird.move()

# Usage
duck = Duck()
ostrich = Ostrich()

print(make_bird_move(duck))    # Output: Flying
print(make_bird_move(ostrich)) # Output: Running

```

Benefits of the Refactor

1. **Correct Substitution**: Both `Duck` and `Ostrich` can now be substituted wherever `Bird` is expected without causing errors. The `move` method is implemented differently, but both subclasses still fulfill the contract expected of a `Bird`.
2. **Behavior Consistency**: The behavior of each subclass (`Duck`, `Ostrich`) aligns with the expectations of the `Bird` superclass, ensuring that the program's correctness is maintained.
3. **Clear Design**: The refactored design separates the concept of `FlyingBird` from `Bird`, making it clear that not all birds can fly, which aligns with real-world scenarios and avoids misusing inheritance.

### Interface Segregation Principle (ISP)

Example that Breaks the Interface Segregation Principle (ISP)

Here’s a Python example that violates the Interface Segregation Principle:

```python
class Printer:
    def print_document(self, document):
        pass

    def scan_document(self, document):
        pass

    def fax_document(self, document):
        pass

class BasicPrinter(Printer):
    def print_document(self, document):
        print(f"Printing: {document}")

    def scan_document(self, document):
        raise NotImplementedError("This printer cannot scan")

    def fax_document(self, document):
        raise NotImplementedError("This printer cannot fax")

class MultiFunctionPrinter(Printer):
    def print_document(self, document):
        print(f"Printing: {document}")

    def scan_document(self, document):
        print(f"Scanning: {document}")

    def fax_document(self, document):
        print(f"Faxing: {document}")

# Usage
basic_printer = BasicPrinter()
basic_printer.print_document("My Document")
basic_printer.scan_document("My Document")  # Raises NotImplementedError

```

Issues with the Code

- The `BasicPrinter` class is forced to implement `scan_document` and `fax_document` methods that it doesn’t actually need. This breaks the Interface Segregation Principle, as the class depends on methods that it does not use.
- This leads to unnecessary code and potential runtime errors, like the `NotImplementedError`.

Refactoring to Follow ISP

To adhere to the Interface Segregation Principle, you can refactor the code by splitting the large `Printer` interface into smaller, more specific interfaces:

```python
from abc import ABC, abstractmethod

class Printable(ABC):
    @abstractmethod
    def print_document(self, document):
        pass

class Scannable(ABC):
    @abstractmethod
    def scan_document(self, document):
        pass

class Faxable(ABC):
    @abstractmethod
    def fax_document(self, document):
        pass

class BasicPrinter(Printable):
    def print_document(self, document):
        print(f"Printing: {document}")

class MultiFunctionPrinter(Printable, Scannable, Faxable):
    def print_document(self, document):
        print(f"Printing: {document}")

    def scan_document(self, document):
        print(f"Scanning: {document}")

    def fax_document(self, document):
        print(f"Faxing: {document}")

# Usage
basic_printer = BasicPrinter()
basic_printer.print_document("My Document")  # Works fine, no unnecessary methods

multi_printer = MultiFunctionPrinter()
multi_printer.print_document("My Document")  # Works fine
multi_printer.scan_document("My Document")   # Works fine

```

Benefits of the Refactor

1. **Specific Interfaces**: Now, `Printable`, `Scannable`, and `Faxable` are specific interfaces. Classes implement only the interfaces they need, so `BasicPrinter` only needs to implement `Printable`.
2. **Reduced Coupling**: The refactor reduces the coupling between classes and interfaces. Classes aren't burdened with methods they don’t use, making the codebase cleaner and less error-prone.
3. **Flexibility**: This design is more flexible because it allows for easy extension. If you need a printer that only prints and scans, you can implement `Printable` and `Scannable` without needing to include `Faxable`.

### Dependency Inversion Principle (DIP)

Example that Breaks the Dependency Inversion Principle (DIP)

Here’s a Python example that violates the Dependency Inversion Principle:

```python
class MySQLDatabase:
    def connect(self):
        print("Connecting to MySQL database...")

    def execute_query(self, query):
        print(f"Executing query on MySQL: {query}")

class UserService:
    def __init__(self):
        self.database = MySQLDatabase()  # Direct dependency on a low-level module

    def get_user(self, user_id):
        self.database.connect()
        return self.database.execute_query(f"SELECT * FROM users WHERE id = {user_id}")

# Usage
service = UserService()
service.get_user(1)

```

Issues with the Code

- The `UserService` class is directly dependent on the `MySQLDatabase` class, which is a low-level module. This creates tight coupling, meaning if you want to switch to a different database (e.g., PostgreSQL), you have to modify the `UserService` class.
- The high-level module (`UserService`) depends on a low-level module (`MySQLDatabase`), violating the Dependency Inversion Principle.

Refactoring to Follow DIP

To adhere to the Dependency Inversion Principle, you can refactor the code to depend on an abstraction (e.g., an interface) rather than a concrete implementation:

```python
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def execute_query(self, query):
        pass

class MySQLDatabase(Database):
    def connect(self):
        print("Connecting to MySQL database...")

    def execute_query(self, query):
        print(f"Executing query on MySQL: {query}")
        return {"user_id": 1, "name": "John Doe"}

class PostgreSQLDatabase(Database):
    def connect(self):
        print("Connecting to PostgreSQL database...")

    def execute_query(self, query):
        print(f"Executing query on PostgreSQL: {query}")
        return {"user_id": 1, "name": "John Doe"}

class UserService:
    def __init__(self, database: Database):  # Depend on abstraction, not a concrete class
        self.database = database

    def get_user(self, user_id):
        self.database.connect()
        return self.database.execute_query(f"SELECT * FROM users WHERE id = {user_id}")

# Usage
mysql_db = MySQLDatabase()
service = UserService(mysql_db)
print(service.get_user(1))

# Easily switch to PostgreSQL without changing UserService
postgres_db = PostgreSQLDatabase()
service_postgres = UserService(postgres_db)
print(service_postgres.get_user(1))

```

Benefits of the Refactor

1. **Reduced Coupling**: The `UserService` class now depends on the `Database` abstraction instead of a concrete implementation like `MySQLDatabase`. This reduces the coupling between high-level and low-level modules.
2. **Flexibility**: The system is more flexible, allowing you to switch database implementations (e.g., from MySQL to PostgreSQL) without modifying the `UserService` class.
3. **Easier Maintenance**: The system is easier to maintain and extend. Adding a new database type involves implementing the `Database` interface, without affecting the rest of the system.
4. **Better Testing**: You can easily mock or stub the `Database` interface for unit testing `UserService`, making tests more isolated and reliable.

### Inheritance Order

**Question:**

Consider the following Python classes:

```python
class A:
    def greet(self):
        return "Hello from A!"

class B(A):
    def greet(self):
        return "Hello from B!"

class C(A):
    def greet(self):
        return "Hello from C!"

class D(B, C):
    pass

```

1. What will be the output of the following code?

```python
d = D()
print(d.greet())

```

1. Explain how Python determines which `greet()` method to call in this case.
2. What is the method resolution order (MRO) for class `D`?
3. How can you explicitly check the MRO of a class in Python?

**Expected Answers:**

1. **Output:**
    
    The output of the code will be:
    
    ```python
    Hello from B!
    
    ```
    
2. **Explanation:**
    
    Python uses the Method Resolution Order (MRO) to determine which method to call in cases of multiple inheritance. The MRO follows the C3 linearization algorithm, which ensures that the method from the first parent class in the inheritance list is chosen. In this case, class `D` inherits from `B` and `C`, so Python looks at class `B` first, finds the `greet()` method there, and calls it.
    
3. **MRO for class `D`:**
    
    The MRO for class `D` is: `[D, B, C, A, object]`.
    
    This means Python will first look for the method in `D`, then `B`, then `C`, then `A`, and finally in the base `object` class if needed.
    
4. **Check MRO:**
    
    You can explicitly check the MRO of a class in Python using the `__mro__` attribute or the `mro()` method:
    
    ```python
    print(D.__mro__)
    # or
    print(D.mro())
    
    ```
    
    This will return:
    
    ```python
    (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
    
    ```
    

This question tests the candidate's understanding of multiple inheritance in Python, particularly how Python resolves method calls when multiple parent classes are involved. It also assesses their familiarity with the Method Resolution Order (MRO) and how to work with it.

## Structure patterns

### Singleton Pattern

**Question:**

Consider the following Python code for implementing a Singleton pattern:

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, value):
        self.value = value

# Create two Singleton instances
singleton1 = Singleton("First")
singleton2 = Singleton("Second")

print(singleton1.value)
print(singleton2.value)

```

1. **What will be the output of the code above?**
2. **Explain how the Singleton pattern is implemented in the provided code.**
3. **What is the purpose of the `__new__` method in the Singleton pattern implementation?**
4. **If you were to modify the `Singleton` class to store multiple instances instead of a single instance, what changes would you make to the `__new__` method?**
5. **What potential issues could arise with this Singleton implementation in a multi-threaded environment, and how would you address them?**

**Expected Answers:**

1. **Output:**
    
    The output of the code will be:
    
    ```
    First
    First
    
    ```
    
    Both `singleton1` and `singleton2` share the same instance, so they both have the same value.
    
2. **Explanation of Implementation:**
    - The `Singleton` class implements the Singleton pattern by ensuring that only one instance of the class is created.
    - The `__new__` method is used to control the creation of a new instance. It checks if `_instance` is `None` (indicating that no instance has been created yet). If `_instance` is `None`, it creates a new instance and assigns it to `_instance`. Otherwise, it returns the existing `_instance`.
3. **Purpose of `__new__` Method:**
    
    The `__new__` method is responsible for creating and returning a new instance of the class. In the Singleton pattern, `__new__` ensures that only one instance of the class is created and reused. The `__init__` method is used to initialize the instance but is not responsible for controlling instance creation.
    
4. **Modifications for Multiple Instances:**
    
    To allow storing multiple instances, you could modify the `__new__` method to manage a collection of instances. For example:
    
    ```python
    class Singleton:
        _instances = {}
    
        def __new__(cls, key, *args, **kwargs):
            if key not in cls._instances:
                cls._instances[key] = super(Singleton, cls).__new__(cls, *args, **kwargs)
            return cls._instances[key]
    
        def __init__(self, value):
            self.value = value
    
    ```
    
    In this modified version, `key` is used to manage multiple instances, and `_instances` is a dictionary that holds these instances.
    
5. **Potential Issues in Multi-threaded Environment:**
    
    In a multi-threaded environment, the current Singleton implementation might create multiple instances if multiple threads enter the `__new__` method simultaneously and find `_instance` to be `None`. To address this, you can use a locking mechanism to ensure thread safety:
    
    ```python
    import threading
    
    class Singleton:
        _instance = None
        _lock = threading.Lock()
    
        def __new__(cls, *args, **kwargs):
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
            return cls._instance
    
        def __init__(self, value):
            self.value = value
    
    ```
    
    Using `threading.Lock`, you can prevent multiple threads from creating separate instances simultaneously.
    

This question assesses the candidate's understanding of the Singleton pattern, including its implementation, purpose of `__new__`, potential modifications, and considerations for multi-threaded environments.

### 1. **Singleton Pattern**

- **Purpose:** Ensures that a class has only one instance and provides a global point of access to that instance.
- **Usage Example:**
    
    ```python
    class Singleton:
        _instance = None
    
        def __new__(cls, *args, **kwargs):
            if not cls._instance:
                cls._instance = super().__new__(cls, *args, **kwargs)
            return cls._instance
    
    ```
    

### 2. **Factory Method Pattern**

- **Purpose:** Defines an interface for creating objects, but allows subclasses to alter the type of objects that will be created.
- **Usage Example:**
    
    ```python
    class Dog:
        def speak(self):
            return "Woof!"
    
    class Cat:
        def speak(self):
            return "Meow!"
    
    class AnimalFactory:
        def get_animal(self, animal_type):
            if animal_type == "Dog":
                return Dog()
            elif animal_type == "Cat":
                return Cat()
    
    ```
    

**Definition:**
The Factory Method Pattern defines an interface for creating objects but allows subclasses to alter the type of objects that will be created. It provides a way to delegate the instantiation logic to subclasses, allowing for more flexible and scalable code.

**Key Concepts:**

1. **Creator Class (or Factory Interface):** Declares the factory method, which returns an object of type `Product`. This is an abstract or base class.
2. **Concrete Creator Class:** Implements the factory method to return an instance of a `ConcreteProduct`.
3. **Product Interface (or Abstract Product):** Defines the interface for the objects that the factory method will create.
4. **Concrete Product Classes:** Implement the `Product` interface and define the specific object to be created.

**Structure:**

- **Product Interface:** Specifies the interface that all concrete products must implement.
- **ConcreteProduct Classes:** Implement the `Product` interface, defining specific behaviors for the objects created by the factory method.
- **Creator (or Factory) Interface:** Declares the factory method, which will return a `Product` instance.
- **ConcreteCreator Classes:** Implement the factory method to create and return an instance of `ConcreteProduct`.

**Example:**

Let's consider a scenario where we need to create different types of documents (e.g., WordDocument and PDFDocument) based on the user’s choice.

**Product Interface:**

```python
from abc import ABC, abstractmethod

class Document(ABC):
    @abstractmethod
    def render(self):
        pass

```

**Concrete Products:**

```python
class WordDocument(Document):
    def render(self):
        return "Rendering a Word document"

class PDFDocument(Document):
    def render(self):
        return "Rendering a PDF document"

```

**Creator Interface:**

```python
class DocumentCreator(ABC):
    @abstractmethod
    def create_document(self) -> Document:
        pass

```

**Concrete Creators:**

```python
class WordDocumentCreator(DocumentCreator):
    def create_document(self) -> Document:
        return WordDocument()

class PDFDocumentCreator(DocumentCreator):
    def create_document(self) -> Document:
        return PDFDocument()

```

**Client Code:**

```python
def client_code(creator: DocumentCreator):
    document = creator.create_document()
    print(document.render())

# Usage
word_creator = WordDocumentCreator()
client_code(word_creator)  # Output: Rendering a Word document

pdf_creator = PDFDocumentCreator()
client_code(pdf_creator)  # Output: Rendering a PDF document

```

**Benefits:**

1. **Encapsulation of Object Creation:** The creation logic is encapsulated within the creator classes. This allows changes to object creation logic without modifying client code.
2. **Decoupling of Client Code:** The client code depends on the `DocumentCreator` interface, not on the concrete implementations of the documents.
3. **Flexibility:** New types of documents can be added by creating new concrete products and creators without changing existing code.

**When to Use:**

- When a class cannot anticipate the type of objects it needs to create.
- When a class wants its subclasses to specify the objects it creates.
- When the creation of objects involves complex logic that should be separated from the code that uses the objects.

The Factory Method Pattern helps in designing systems that are easy to extend and maintain by isolating the object creation process from the code that uses the objects.

### 3. **Abstract Factory Pattern**

- **Purpose:** Provides an interface for creating families of related or dependent objects without specifying their concrete classes.
- **Usage Example:**
    
    ```python
    class Dog:
        def speak(self):
            return "Woof!"
    
    class Cat:
        def speak(self):
            return "Meow!"
    
    class AnimalFactory:
        def create_dog(self):
            pass
        def create_cat(self):
            pass
    
    class ConcreteAnimalFactory(AnimalFactory):
        def create_dog(self):
            return Dog()
    
        def create_cat(self):
            return Cat()
    
    ```
    

**Definition:**
The Abstract Factory Pattern provides an interface for creating families of related or dependent objects without specifying their concrete classes. It allows for the creation of multiple types of objects (usually grouped into families) through a factory interface, ensuring that related objects are created in a consistent way.

**Key Concepts:**

1. **Abstract Factory Interface:** Declares methods for creating abstract products. It typically includes methods for creating different types of products that belong to the same family.
2. **Concrete Factories:** Implement the abstract factory interface to produce concrete products.
3. **Abstract Products:** Define interfaces for a set of related or dependent objects. Each abstract product interface represents a single type of object.
4. **Concrete Products:** Implement the abstract product interfaces. Each concrete product corresponds to a specific variant of a product in the family.
5. **Client:** Uses the abstract factory and abstract product interfaces to interact with the created objects. The client does not need to know the concrete classes of the objects it uses.

**Structure:**

- **AbstractFactory:** Interface with methods for creating abstract products.
- **ConcreteFactory:** Implements the abstract factory interface to create concrete products.
- **AbstractProductA:** Interface for the first type of product.
- **ConcreteProductA1:** Implements `AbstractProductA`.
- **ConcreteProductA2:** Implements `AbstractProductA`.
- **AbstractProductB:** Interface for the second type of product.
- **ConcreteProductB1:** Implements `AbstractProductB`.
- **ConcreteProductB2:** Implements `AbstractProductB`.
- **Client:** Uses the abstract factory to create and interact with products.

**Example:**

Consider a scenario where we need to create user interfaces for different operating systems. Each operating system has its own style of buttons and dialogs. We want to create a consistent UI for each OS without coupling the client code to specific implementations.

**Abstract Factory Interface:**

```python
from abc import ABC, abstractmethod

class GUIFactory(ABC):
    @abstractmethod
    def create_button(self):
        pass

    @abstractmethod
    def create_dialog(self):
        pass

```

**Abstract Products:**

```python
class Button(ABC):
    @abstractmethod
    def render(self):
        pass

class Dialog(ABC):
    @abstractmethod
    def show(self):
        pass

```

**Concrete Products for Windows:**

```python
class WindowsButton(Button):
    def render(self):
        return "Rendering a Windows button"

class WindowsDialog(Dialog):
    def show(self):
        return "Showing a Windows dialog"

```

**Concrete Products for macOS:**

```python
class MacButton(Button):
    def render(self):
        return "Rendering a Mac button"

class MacDialog(Dialog):
    def show(self):
        return "Showing a Mac dialog"

```

**Concrete Factories:**

```python
class WindowsFactory(GUIFactory):
    def create_button(self):
        return WindowsButton()

    def create_dialog(self):
        return WindowsDialog()

class MacFactory(GUIFactory):
    def create_button(self):
        return MacButton()

    def create_dialog(self):
        return MacDialog()

```

**Client Code:**

```python
def client_code(factory: GUIFactory):
    button = factory.create_button()
    dialog = factory.create_dialog()
    print(button.render())
    print(dialog.show())

# Usage
windows_factory = WindowsFactory()
client_code(windows_factory)

mac_factory = MacFactory()
client_code(mac_factory)

```

**Benefits:**

1. **Consistency:** Ensures that products from the same family are created in a consistent manner.
2. **Encapsulation:** Encapsulates the creation logic of related products, making the code more modular and easier to extend.
3. **Decoupling:** Decouples the client code from the concrete classes of the products, allowing for flexibility and easier maintenance.
4. **Scalability:** New product families can be added with minimal changes to the existing codebase. You only need to create new concrete factories and products.

**When to Use:**

- When a system needs to create multiple families of related objects and the client code should not depend on the specific classes of these objects.
- When you want to provide a library or framework that works with multiple types of products but doesn't require the user to understand the specific product types.

The Abstract Factory Pattern is ideal for scenarios where a system needs to work with various sets of related objects, and you want to ensure that these objects are created in a consistent way without hardcoding the details of their creation.

### 4. **Builder Pattern**

- **Purpose:** Separates the construction of a complex object from its representation, allowing the same construction process to create different representations.
- **Usage Example:**
    
    ```python
    class Car:
        def __init__(self):
            self.model = None
            self.color = None
    
    class CarBuilder:
        def __init__(self):
            self.car = Car()
    
        def set_model(self, model):
            self.car.model = model
            return self
    
        def set_color(self, color):
            self.car.color = color
            return self
    
        def build(self):
            return self.car
    
    ```
    

**Definition:**
The Builder Pattern is a design pattern that provides a way to construct a complex object step by step. It separates the construction of a complex object from its representation, allowing the same construction process to create different representations. This pattern is particularly useful for constructing objects with many optional components or configurations.

**Key Concepts:**

1. **Builder:** An abstract interface for creating parts of a complex object. It declares methods for creating each component of the product.
2. **ConcreteBuilder:** Implements the builder interface to construct and assemble parts of the product. It defines the specific steps for building the product.
3. **Director:** Constructs the object using the builder interface. It defines the order in which to call the builder's methods to produce a specific configuration of the product.
4. **Product:** The complex object that is being constructed. It typically has multiple parts or components that are assembled together.

**Structure:**

- **Builder:** Interface with methods for creating and assembling different parts of the product.
- **ConcreteBuilder:** Implements the `Builder` interface to build and assemble the product. It holds the final product instance.
- **Director:** Uses a `Builder` instance to construct the product step by step. It controls the construction process.
- **Product:** Represents the complex object that is being constructed. It includes various components or parts.

**Example:**

Consider a scenario where we need to construct a complex object like a `Car` with various optional features such as a sunroof, leather seats, and a navigation system. The `Car` can be built in different configurations depending on the features chosen.

**Builder Interface:**

```python
from abc import ABC, abstractmethod

class CarBuilder(ABC):
    @abstractmethod
    def set_engine(self, engine: str):
        pass

    @abstractmethod
    def set_wheels(self, wheels: int):
        pass

    @abstractmethod
    def set_color(self, color: str):
        pass

    @abstractmethod
    def set_sunroof(self, has_sunroof: bool):
        pass

    @abstractmethod
    def set_navigation_system(self, has_navigation: bool):
        pass

    @abstractmethod
    def get_result(self):
        pass

```

**ConcreteBuilder:**

```python
class SportsCarBuilder(CarBuilder):
    def __init__(self):
        self.reset()

    def reset(self):
        self._car = Car()

    def set_engine(self, engine: str):
        self._car.engine = engine

    def set_wheels(self, wheels: int):
        self._car.wheels = wheels

    def set_color(self, color: str):
        self._car.color = color

    def set_sunroof(self, has_sunroof: bool):
        self._car.sunroof = has_sunroof

    def set_navigation_system(self, has_navigation: bool):
        self._car.navigation_system = has_navigation

    def get_result(self):
        car = self._car
        self.reset()
        return car

```

**Director:**

```python
class CarDirector:
    def __init__(self, builder: CarBuilder):
        self._builder = builder

    def construct_sports_car(self):
        self._builder.set_engine("V8")
        self._builder.set_wheels(4)
        self._builder.set_color("Red")
        self._builder.set_sunroof(True)
        self._builder.set_navigation_system(True)

```

**Product:**

```python
class Car:
    def __init__(self):
        self.engine = None
        self.wheels = None
        self.color = None
        self.sunroof = None
        self.navigation_system = None

    def __str__(self):
        return (f"Car with engine: {self.engine}, wheels: {self.wheels}, color: {self.color}, "
                f"sunroof: {self.sunroof}, navigation system: {self.navigation_system}")

```

**Client Code:**

```python
# Create a builder and director
builder = SportsCarBuilder()
director = CarDirector(builder)

# Construct a sports car
director.construct_sports_car()
car = builder.get_result()

print(car)

```

**Benefits:**

1. **Separation of Concerns:** Separates the construction of a complex object from its representation. This allows for more flexible and maintainable code.
2. **Immutability:** Allows for the creation of immutable objects by providing a clear separation between the building process and the final product.
3. **Controlled Construction:** Ensures that the product is constructed in a consistent and controlled manner, following the steps defined by the director.
4. **Customization:** Provides a way to build complex objects with varying configurations, making it easy to create different variants of the product.

**When to Use:**

- When an object needs to be constructed with many optional components or configurations.
- When you want to separate the construction logic from the final object representation.
- When you want to create different variations of a product using a common construction process.

The Builder Pattern is ideal for scenarios where you need to construct complex objects step by step, allowing for customization and variation in the final product while keeping the construction process organized and flexible.

### 5. **Prototype Pattern**

- **Purpose:** Creates new objects by copying an existing object, known as the prototype.
- **Usage Example:**
    
    ```python
    import copy
    
    class Prototype:
        def __init__(self):
            self._objects = {}
    
        def register_object(self, name, obj):
            self._objects[name] = obj
    
        def unregister_object(self, name):
            del self._objects[name]
    
        def clone(self, name, **attrs):
            obj = copy.deepcopy(self._objects.get(name))
            obj.__dict__.update(attrs)
            return obj
    
    ```
    

### 6. **Decorator Pattern**

- **Purpose:** Adds behavior to objects dynamically without affecting the behavior of other objects from the same class.
- **Usage Example:**
    
    ```python
    class Coffee:
        def cost(self):
            return 5
    
    class MilkDecorator:
        def __init__(self, coffee):
            self._coffee = coffee
    
        def cost(self):
            return self._coffee.cost() + 2
    
    ```
    

**Definition:**
The Decorator Pattern is a structural design pattern that allows behavior to be added to individual objects, either statically or dynamically, without affecting the behavior of other objects from the same class. It provides a flexible alternative to subclassing for extending functionality.

**Key Concepts:**

1. **Component:** The interface or abstract class that defines the operations that can be dynamically added to concrete implementations. It usually declares methods that the concrete components and decorators will implement.
2. **ConcreteComponent:** The class that implements the `Component` interface. This is the base class that can have additional behavior added to it.
3. **Decorator:** An abstract class or interface that also implements the `Component` interface. It contains a reference to a `Component` object and can delegate operations to this object. It also defines a way to add extra functionality.
4. **ConcreteDecorator:** A class that extends the `Decorator` class and adds additional behavior to the `ConcreteComponent`. Each concrete decorator can add its own specific behavior.

**Structure:**

- **Component:** Defines the interface for objects that can have responsibilities added to them.
- **ConcreteComponent:** Implements the `Component` interface and defines the basic functionality that can be decorated.
- **Decorator:** Implements the `Component` interface and maintains a reference to a `Component` object. It delegates operations to the `Component` and adds additional behavior.
- **ConcreteDecorator:** Extends the `Decorator` class and implements additional behaviors.

**Example:**

Consider a scenario where we have a basic `Coffee` class, and we want to add different features to it, such as milk, sugar, or whipped cream, without modifying the original `Coffee` class.

**Component Interface:**

```python
from abc import ABC, abstractmethod

class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float:
        pass

```

**ConcreteComponent:**

```python
class BasicCoffee(Coffee):
    def cost(self) -> float:
        return 5.0

```

**Decorator Interface:**

```python
class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee

    @abstractmethod
    def cost(self) -> float:
        return self._coffee.cost()

```

**ConcreteDecorators:**

```python
class MilkDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 1.0

class SugarDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.5

class WhippedCreamDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 2.0

```

**Client Code:**

```python
# Create a basic coffee
basic_coffee = BasicCoffee()

# Decorate the coffee with milk and sugar
milk_coffee = MilkDecorator(basic_coffee)
sugar_milk_coffee = SugarDecorator(milk_coffee)
whipped_cream_sugar_milk_coffee = WhippedCreamDecorator(sugar_milk_coffee)

# Get the final cost
print(f"Cost of coffee: ${whipped_cream_sugar_milk_coffee.cost():.2f}")

```

**Benefits:**

1. **Flexibility:** Allows behavior to be added or extended at runtime. Multiple decorators can be combined to achieve different behaviors.
2. **Open/Closed Principle:** Follows the Open/Closed Principle by allowing new functionality to be added without modifying existing code.
3. **Single Responsibility Principle:** Each decorator focuses on a single responsibility, making it easier to understand and maintain.
4. **Dynamic Behavior Addition:** Supports dynamic behavior addition and combination, allowing for versatile and reusable code.

**When to Use:**

- When you want to add new responsibilities to objects without affecting other objects.
- When you need to combine multiple behaviors in a flexible and reusable manner.
- When subclassing becomes impractical due to the large number of combinations of behaviors.

The Decorator Pattern is ideal for scenarios where you need to extend the behavior of objects in a flexible and modular way, avoiding the complexity of subclassing and allowing for the dynamic combination of features.

### 7. **Adapter Pattern**

- **Purpose:** Allows incompatible interfaces to work together by converting the interface of a class into another interface that a client expects.
- **Usage Example:**
    
    ```python
    class OldSystem:
        def old_method(self):
            return "Old method"
    
    class NewSystem:
        def new_method(self):
            return "New method"
    
    class Adapter:
        def __init__(self, new_system):
            self._new_system = new_system
    
        def old_method(self):
            return self._new_system.new_method()
    
    ```
    

**Definition:**
The Adapter Pattern is a structural design pattern that allows incompatible interfaces to work together. It acts as a bridge between two incompatible interfaces by wrapping one of the interfaces and converting its methods into a format that the other interface can understand.

**Key Concepts:**

1. **Target:** The interface that the client code expects to use. It defines the domain-specific interface that is used by the client.
2. **Adaptee:** The existing class or interface that has an incompatible interface with the `Target`. It contains methods that need to be adapted to meet the `Target` interface.
3. **Adapter:** A class that implements the `Target` interface and translates the requests from the `Target` to the methods of the `Adaptee`. It holds an instance of the `Adaptee` and delegates the calls to it, adapting the calls as necessary.

**Structure:**

- **Target:** Defines the interface that the client code uses.
- **Adaptee:** Defines an existing interface with methods that need to be adapted.
- **Adapter:** Implements the `Target` interface and adapts the methods of the `Adaptee` to the `Target` interface.

**Example:**

Consider a scenario where we have an application that expects a `Rectangle` interface, but we have an existing `LegacySquare` class that only provides a `get_side_length` method.

**Target Interface:**

```python
class Rectangle:
    def get_width(self) -> float:
        pass

    def get_height(self) -> float:
        pass

```

**Adaptee Class:**

```python
class LegacySquare:
    def __init__(self, side_length: float):
        self.side_length = side_length

    def get_side_length(self) -> float:
        return self.side_length

```

**Adapter Class:**

```python
class SquareAdapter(Rectangle):
    def __init__(self, square: LegacySquare):
        self._square = square

    def get_width(self) -> float:
        return self._square.get_side_length()

    def get_height(self) -> float:
        return self._square.get_side_length()

```

**Client Code:**

```python
def print_rectangle_dimensions(rectangle: Rectangle):
    print(f"Width: {rectangle.get_width()}, Height: {rectangle.get_height()}")

# Create a LegacySquare object
legacy_square = LegacySquare(side_length=5.0)

# Create an adapter for the LegacySquare
adapter = SquareAdapter(legacy_square)

# Use the adapter as a Rectangle
print_rectangle_dimensions(adapter)

```

**Benefits:**

1. **Compatibility:** Allows classes with incompatible interfaces to work together without modifying their code.
2. **Single Responsibility Principle:** Adheres to this principle by separating the adaptation logic from the main class or client code.
3. **Flexibility:** Provides a way to extend and integrate existing systems with minimal changes.
4. **Reusability:** Facilitates the reuse of existing code by adapting it to new interfaces.

**When to Use:**

- When you need to integrate a new system or library with an existing codebase that uses a different interface.
- When you want to use third-party classes that have incompatible interfaces with your existing code.
- When you need to provide a consistent interface for clients while dealing with legacy or external systems.

The Adapter Pattern is useful for enabling interoperability between disparate systems, allowing different parts of a system to work together seamlessly, and providing a flexible solution for integrating and adapting existing code.

### 8. **Proxy Pattern**

- **Purpose:** Provides a surrogate or placeholder for another object to control access to it.
- **Usage Example:**
    
    ```python
    class RealSubject:
        def request(self):
            return "Real Subject Request"
    
    class Proxy:
        def __init__(self, real_subject):
            self._real_subject = real_subject
    
        def request(self):
            # Add additional functionality here
            return self._real_subject.request()
    
    ```
    

These patterns help address common problems in software design and promote code reuse, flexibility, and maintainability.

**Definition:**
The Proxy Pattern is a structural design pattern where a surrogate or placeholder object (the proxy) controls access to another object. It acts as an intermediary, allowing you to manage the interactions with the real object, often to add additional functionality like access control, lazy initialization, logging, or other enhancements.

**Key Concepts:**

1. **Subject:** Defines the common interface for RealSubject and Proxy. It provides the methods that both RealSubject and Proxy must implement.
2. **RealSubject:** The actual object that the proxy is controlling access to. It implements the Subject interface and contains the core business logic or functionality.
3. **Proxy:** A class that implements the Subject interface and holds a reference to the RealSubject. It manages access to the RealSubject, adding additional behavior as necessary.

**Types of Proxies:**

1. **Virtual Proxy:** Delays the creation of a resource-intensive object until it is actually needed. This is useful for managing memory or performance.
2. **Remote Proxy:** Represents an object that is located on a different address space (such as a different server). It provides a way to interact with the remote object as if it were local.
3. **Protection Proxy:** Controls access to the RealSubject, often implementing security checks or access control mechanisms. It ensures that only authorized clients can access certain functionality.
4. **Cache Proxy:** Manages a cache of the RealSubject’s data to improve performance by avoiding repeated expensive operations.

**Structure:**

- **Subject Interface:** Provides the common interface for RealSubject and Proxy.
- **RealSubject:** Implements the Subject interface and contains the actual business logic.
- **Proxy:** Implements the Subject interface, holds a reference to the RealSubject, and controls access to it.

**Example:**

Consider a scenario where you have a resource-intensive `Image` object that you want to load and display. You can use a Proxy to delay the loading of the image until it is actually needed.

**Subject Interface:**

```python
class Image:
    def display(self) -> None:
        pass

```

**RealSubject Class:**

```python
class RealImage(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self.load_image()

    def load_image(self) -> None:
        print(f"Loading image: {self.filename}")

    def display(self) -> None:
        print(f"Displaying image: {self.filename}")

```

**Proxy Class:**

```python
class ProxyImage(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self.real_image = None

    def display(self) -> None:
        if self.real_image is None:
            self.real_image = RealImage(self.filename)
        self.real_image.display()

```

**Client Code:**

```python
def show_image(image: Image) -> None:
    image.display()

# Create a ProxyImage object
proxy_image = ProxyImage("large_photo.jpg")

# Display the image
show_image(proxy_image)  # Loads and displays the image
show_image(proxy_image)  # Displays the already loaded image

```

**Benefits:**

1. **Lazy Initialization:** Delays the creation of resource-intensive objects until they are actually needed, saving resources and improving performance.
2. **Access Control:** Provides a way to manage access to the RealSubject, enforcing security or permission checks.
3. **Logging and Monitoring:** Allows for additional functionality such as logging, monitoring, or tracking interactions with the RealSubject.
4. **Performance Optimization:** Can implement caching or other optimizations to improve performance and reduce redundant operations.

**When to Use:**

- When you need to control access to a resource-intensive or expensive object.
- When you want to delay the initialization of an object until it is actually needed.
- When you need to manage access control, logging, or other cross-cutting concerns.
- When you want to provide a proxy to interact with remote objects or services.

The Proxy Pattern is useful for managing interactions with objects that are costly to create or access, providing additional functionality such as caching, access control, and performance optimization, while keeping the client code simple and clean.

### 9. Facade Pattern

**Definition:**
The Facade Pattern is a structural design pattern that provides a simplified, unified interface to a set of interfaces in a subsystem. It defines a higher-level interface that makes the subsystem easier to use. The facade acts as a bridge between the client code and the complex subsystem, hiding the complexities and providing a simpler interface for interacting with the subsystem.

**Key Concepts:**

1. **Facade:** A class that provides a simplified interface to a complex subsystem. It delegates requests from the client to the appropriate objects within the subsystem, making the subsystem easier to use.
2. **Subsystem Classes:** The classes that make up the subsystem. These classes are often complex and have multiple interfaces. The Facade class interacts with these classes to perform various tasks.

**Structure:**

- **Facade:** Implements a simplified interface that the client uses. It delegates calls to the appropriate classes in the subsystem.
- **Subsystem Classes:** Encapsulate the complex functionality of the subsystem. They interact with each other to perform the actual work but may have complex and numerous interfaces.

**Example:**

Consider a home theater system with various components like a DVD player, projector, and sound system. Each component has a complex interface, but you want to provide a simple interface for users to operate the entire home theater system.

**Subsystem Classes:**

```python
class DVDPlayer:
    def on(self):
        print("DVD Player is now ON")

    def play(self, movie: str):
        print(f"Playing movie: {movie}")

    def off(self):
        print("DVD Player is now OFF")

class Projector:
    def on(self):
        print("Projector is now ON")

    def set_input(self, input_source: str):
        print(f"Projector input set to: {input_source}")

    def off(self):
        print("Projector is now OFF")

class SoundSystem:
    def on(self):
        print("Sound System is now ON")

    def set_volume(self, volume: int):
        print(f"Sound System volume set to: {volume}")

    def off(self):
        print("Sound System is now OFF")

```

**Facade Class:**

```python
class HomeTheaterFacade:
    def __init__(self, dvd_player: DVDPlayer, projector: Projector, sound_system: SoundSystem):
        self.dvd_player = dvd_player
        self.projector = projector
        self.sound_system = sound_system

    def watch_movie(self, movie: str) -> None:
        print("Preparing to watch a movie...")
        self.projector.on()
        self.projector.set_input("DVD")
        self.sound_system.on()
        self.sound_system.set_volume(10)
        self.dvd_player.on()
        self.dvd_player.play(movie)

    def end_movie(self) -> None:
        print("Turning off the home theater...")
        self.dvd_player.off()
        self.sound_system.off()
        self.projector.off()

```

**Client Code:**

```python
def main():
    dvd_player = DVDPlayer()
    projector = Projector()
    sound_system = SoundSystem()
    home_theater = HomeTheaterFacade(dvd_player, projector, sound_system)

    # Use the Facade to watch a movie
    home_theater.watch_movie("Inception")

    # Use the Facade to end the movie
    home_theater.end_movie()

if __name__ == "__main__":
    main()

```

**Benefits:**

1. **Simplifies Interface:** Provides a simple and unified interface to a complex subsystem, making it easier for clients to use the subsystem.
2. **Encapsulation:** Hides the complexities of the subsystem from the client code. Changes to the subsystem do not affect the client code if the Facade interface remains consistent.
3. **Reduces Dependencies:** Reduces the number of dependencies between the client code and the subsystem. The client interacts only with the Facade, not directly with the subsystem classes.
4. **Improves Maintainability:** Makes the subsystem easier to maintain by centralizing the interactions with the subsystem in one place.
5. **Enhanced Readability:** Improves the readability and usability of the code by providing a clean and coherent interface for the subsystem functionality.

**When to Use:**

- When you need to provide a simple interface to a complex subsystem or set of classes.
- When you want to decouple client code from the subsystem, reducing the impact of changes in the subsystem.
- When you want to create a unified interface for a set of classes that perform related functions.
- When you want to improve the readability and usability of the code by consolidating complex interactions into a single class.

The Facade Pattern is useful for creating a simple and cohesive interface to a complex system, making it easier for clients to interact with the system without needing to understand its internal complexities.

## Behavioral patterns

Behavioral design patterns focus on how objects interact and collaborate to achieve certain behaviors. They help manage complex control flow and communication between objects. Here’s a list of some of the most common behavioral design patterns:

### 1. **Chain of Responsibility**

**Definition:** Passes a request along a chain of handlers. Each handler can either process the request or pass it to the next handler in the chain.

**Use Case:** When you have multiple objects that can handle a request, and the handler is not known until runtime.

**Example:** Handling HTTP requests in a web server, where each handler (middleware) processes the request or passes it to the next handler.

The Chain of Responsibility pattern is a behavioral design pattern that allows you to pass requests along a chain of handlers. Each handler in the chain either processes the request or passes it to the next handler. This pattern decouples the sender of the request from its receiver, giving more flexibility in assigning responsibilities.

Example Scenario: Logging System

Imagine a logging system where messages can be logged at different levels: `INFO`, `WARNING`, and `ERROR`. Depending on the level, the message should be handled by different loggers. The Chain of Responsibility pattern can be used to create a chain of loggers that can handle or pass along the logging request.

Python Code Example

```python
from abc import ABC, abstractmethod

class Logger(ABC):
    def __init__(self, level):
        self.level = level
        self.next_logger = None

    def set_next_logger(self, next_logger):
        self.next_logger = next_logger

    def log_message(self, level, message):
        if self.level <= level:
            self.write_message(message)
        if self.next_logger:
            self.next_logger.log_message(level, message)

    @abstractmethod
    def write_message(self, message):
        pass

class InfoLogger(Logger):
    def __init__(self, level):
        super().__init__(level)

    def write_message(self, message):
        print(f"[INFO]: {message}")

class WarningLogger(Logger):
    def __init__(self, level):
        super().__init__(level)

    def write_message(self, message):
        print(f"[WARNING]: {message}")

class ErrorLogger(Logger):
    def __init__(self, level):
        super().__init__(level)

    def write_message(self, message):
        print(f"[ERROR]: {message}")

# Setting up the chain of responsibility
info_logger = InfoLogger(1)
warning_logger = WarningLogger(2)
error_logger = ErrorLogger(3)

info_logger.set_next_logger(warning_logger)
warning_logger.set_next_logger(error_logger)

# Client code
info_logger.log_message(1, "This is an informational message.")
info_logger.log_message(2, "This is a warning message.")
info_logger.log_message(3, "This is an error message.")

```

Explanation

1. **Logger Base Class:**
    - The `Logger` class is an abstract base class that defines a method `log_message` to process logging requests.
    - Each logger can be linked to the next logger using the `set_next_logger` method.
    - The `log_message` method checks if the current logger should handle the message (based on its level) and either processes it or passes it to the next logger in the chain.
2. **Concrete Loggers:**
    - `InfoLogger`, `WarningLogger`, and `ErrorLogger` are concrete implementations of the `Logger` class.
    - Each logger defines how to handle the message using the `write_message` method.
3. **Chain Setup:**
    - The chain of responsibility is created by linking loggers together: `info_logger` → `warning_logger` → `error_logger`.
4. **Client Code:**
    - The client code sends a logging request to the first logger in the chain (`info_logger`), which processes it or passes it along to the next logger.

Output

```
[INFO]: This is an informational message.
[WARNING]: This is a warning message.
[ERROR]: This is an error message.

```

Benefits

- **Decoupling:** The sender of the request doesn't need to know which object will handle it.
- **Flexibility:** Handlers can be added or removed dynamically without affecting the client code.
- **Responsibility Distribution:** Multiple handlers can contribute to processing a single request, allowing for shared responsibilities.

This pattern is useful in scenarios where multiple handlers can process a request, and the exact handler isn't known until runtime. It provides a flexible and scalable way to manage complex chains of operations.

### 2. **Command**

**Definition:** Encapsulates a request as an object, thereby allowing parameterization and queuing of requests. It also supports undo operations.

**Use Case:** When you need to issue requests, queue operations, or support undo functionality.

**Example:** Implementing undo/redo functionality in a text editor.

The Command pattern is a behavioral design pattern that turns a request into a stand-alone object containing all information about the request. This transformation allows you to parameterize methods with different requests, delay or queue a request's execution, and support undoable operations.

Example Scenario: Remote Control System

Imagine a remote control system for a smart home where you can turn on/off various devices like lights, fans, and TVs. The Command pattern can be used to encapsulate the actions (like turning on/off a device) into objects, allowing the remote control to execute commands, undo them, or store them for later execution.

Python Code Example

```python
from abc import ABC, abstractmethod

# Command Interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass

# Concrete Command Classes
class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.on()

    def undo(self):
        self.light.off()

class LightOffCommand(Command):
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.off()

    def undo(self):
        self.light.on()

class FanOnCommand(Command):
    def __init__(self, fan):
        self.fan = fan

    def execute(self):
        self.fan.on()

    def undo(self):
        self.fan.off()

class FanOffCommand(Command):
    def __init__(self, fan):
        self.fan = fan

    def execute(self):
        self.fan.off()

    def undo(self):
        self.fan.on()

# Receiver Classes
class Light:
    def on(self):
        print("The light is on")

    def off(self):
        print("The light is off")

class Fan:
    def on(self):
        print("The fan is on")

    def off(self):
        print("The fan is off")

# Invoker Class
class RemoteControl:
    def __init__(self):
        self.history = []

    def execute_command(self, command):
        command.execute()
        self.history.append(command)

    def undo_last_command(self):
        if self.history:
            last_command = self.history.pop()
            last_command.undo()

# Client Code
light = Light()
fan = Fan()

light_on_command = LightOnCommand(light)
light_off_command = LightOffCommand(light)
fan_on_command = FanOnCommand(fan)
fan_off_command = FanOffCommand(fan)

remote = RemoteControl()

# Execute Commands
remote.execute_command(light_on_command)  # The light is on
remote.execute_command(fan_on_command)    # The fan is on

# Undo Commands
remote.undo_last_command()  # The fan is off
remote.undo_last_command()  # The light is off

```

Explanation

1. **Command Interface:**
    - The `Command` class defines the interface for executing and undoing actions. Each concrete command class implements this interface.
2. **Concrete Command Classes:**
    - `LightOnCommand`, `LightOffCommand`, `FanOnCommand`, and `FanOffCommand` are concrete implementations of the `Command` interface. They encapsulate the request to perform an action on the receiver (like turning on/off a light or fan).
3. **Receiver Classes:**
    - `Light` and `Fan` are the classes that contain the actual logic for turning on/off the devices. These classes are the receivers of the command.
4. **Invoker Class:**
    - The `RemoteControl` class acts as the invoker. It stores the history of executed commands and can execute or undo commands.
5. **Client Code:**
    - The client code creates instances of the receivers, concrete commands, and the invoker. It then executes commands via the invoker and can also undo the last executed command.

Output

```
The light is on
The fan is on
The fan is off
The light is off

```

Benefits

- **Decoupling:** The sender (e.g., `RemoteControl`) is decoupled from the receiver (e.g., `Light`, `Fan`). The sender only knows about the command interface, not how the action is performed.
- **Undo/Redo Operations:** Commands can be stored in a history, allowing the ability to undo or redo operations.
- **Flexible Command Handling:** New commands can be added without modifying existing classes, adhering to the Open/Closed Principle.

This pattern is especially useful in scenarios where actions need to be decoupled from the objects that perform them, enabling features like undo/redo, command scheduling, or logging operations.

### 3. **Interpreter**

**Definition:** Defines a grammatical representation for a language and provides an interpreter to interpret sentences in the language.

**Use Case:** When you need to interpret or compile sentences in a language.

**Example:** Implementing a simple expression evaluator or a domain-specific language (DSL).

The Interpreter pattern is a behavioral design pattern that defines a representation for a language's grammar along with an interpreter that uses this representation to interpret sentences in the language. This pattern is useful when you need to interpret expressions from a predefined grammar or language.

Example Scenario: Simple Mathematical Expression Interpreter

Imagine you need to create an interpreter for a very basic mathematical expression language that can handle addition and subtraction of single digits. The expressions are provided as strings, such as "3 + 5 - 2", and you want to evaluate them.

Python Code Example

```python
from abc import ABC, abstractmethod

# Abstract Expression
class Expression(ABC):
    @abstractmethod
    def interpret(self):
        pass

# Terminal Expression for Numbers
class Number(Expression):
    def __init__(self, value):
        self.value = int(value)

    def interpret(self):
        return self.value

# Non-terminal Expression for Additions
class Add(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interpret(self):
        return self.left.interpret() + self.right.interpret()

# Non-terminal Expression for Subtractions
class Subtract(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interpret(self):
        return self.left.interpret() - self.right.interpret()

# Client Code
def parse_expression(expression):
    stack = []
    tokens = expression.split()

    for token in tokens:
        if token.isdigit():
            stack.append(Number(token))
        elif token == '+':
            right = stack.pop()
            left = stack.pop()
            stack.append(Add(left, right))
        elif token == '-':
            right = stack.pop()
            left = stack.pop()
            stack.append(Subtract(left, right))

    return stack.pop()

# Example Usage
expression = "3 + 5 - 2"
interpreter = parse_expression(expression)
result = interpreter.interpret()

print(f"Result of '{expression}' is: {result}")

```

Explanation

1. **Expression Interface:**
    - The `Expression` class is an abstract base class that defines the `interpret` method. This method will be implemented by all concrete expressions.
2. **Terminal Expression:**
    - The `Number` class is a terminal expression that represents the numeric values in the expression. It simply returns its value when `interpret` is called.
3. **Non-terminal Expressions:**
    - The `Add` and `Subtract` classes are non-terminal expressions that represent addition and subtraction operations. They hold references to the left and right operands and return the result of their respective operations when `interpret` is called.
4. **Client Code:**
    - The `parse_expression` function is responsible for parsing the input string and constructing the expression tree. It uses a stack to evaluate the expression based on the tokens.
    - The function processes each token, creating `Number`, `Add`, and `Subtract` objects as needed, and then pushes or pops them from the stack to form the expression tree.
    - Finally, it returns the root of the expression tree.
5. **Example Usage:**
    - The example evaluates the expression `"3 + 5 - 2"`. The `parse_expression` function constructs the corresponding expression tree, and the `interpret` method evaluates it to return the result.

Output

```
Result of '3 + 5 - 2' is: 6

```

Benefits

- **Extensibility:** The grammar of the language can be extended by adding new expression classes (e.g., multiplication, division) without modifying existing code.
- **Readability:** The code becomes easier to understand by breaking down the language's grammar into simple classes.
- **Reusability:** The interpreter classes can be reused across different parts of the system that need to evaluate the same language.

Drawbacks

- **Complexity:** The pattern can lead to complex class hierarchies, especially for more elaborate grammars.
- **Performance:** Interpreters can be slow when dealing with complex expressions, especially if they involve deep recursion.

The Interpreter pattern is particularly useful when you have a simple grammar that needs to be interpreted or evaluated repeatedly. It provides a clear, modular way to define and extend the rules for interpreting expressions.

### 4. **Iterator**

**Definition:** Provides a way to access elements of an aggregate object sequentially without exposing its underlying representation.

**Use Case:** When you need to traverse a collection of objects without exposing the internal structure.

**Example:** Iterating over elements in a collection like lists or trees.

The Iterator pattern is a behavioral design pattern that provides a way to access the elements of a collection (e.g., list, tree, or graph) sequentially without exposing the underlying representation of the collection. It abstracts the process of iterating over a collection, allowing you to use the same interface to traverse different data structures.

Example Scenario: Custom Collection of Books

Imagine you have a collection of books in a library, and you want to iterate over this collection to access each book without exposing how the books are stored internally.

Python Code Example

```python
from collections.abc import Iterator, Iterable

# Book class representing an individual book
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        return f"'{self.title}' by {self.author}"

# Concrete collection class
class Library(Iterable):
    def __init__(self):
        self._books = []

    def add_book(self, book):
        self._books.append(book)

    def __iter__(self):
        return LibraryIterator(self._books)

# Concrete iterator class
class LibraryIterator(Iterator):
    def __init__(self, books):
        self._books = books
        self._index = 0

    def __next__(self):
        if self._index < len(self._books):
            book = self._books[self._index]
            self._index += 1
            return book
        else:
            raise StopIteration

# Client code
library = Library()
library.add_book(Book("1984", "George Orwell"))
library.add_book(Book("To Kill a Mockingbird", "Harper Lee"))
library.add_book(Book("The Great Gatsby", "F. Scott Fitzgerald"))

for book in library:
    print(book)

```

### Explanation

1. **Book Class:**
    - The `Book` class represents an individual book with a title and an author. It's a simple data structure that holds book information.
2. **Library Class (Concrete Collection):**
    - The `Library` class is a concrete collection that holds a list of books (`_books`). It provides methods to add books to the collection and implements the `__iter__` method, which returns an iterator object (`LibraryIterator`).
3. **LibraryIterator Class (Concrete Iterator):**
    - The `LibraryIterator` class is the concrete iterator that implements the `Iterator` interface. It maintains an internal index to keep track of the current position in the collection.
    - The `__next__` method is implemented to return the next book in the collection. If the end of the collection is reached, it raises a `StopIteration` exception, signaling that the iteration is complete.
4. **Client Code:**
    - The client code creates a `Library` instance and adds several `Book` objects to it.
    - It then iterates over the `Library` using a `for` loop, which internally uses the `LibraryIterator` to access each book sequentially.

Output

```
'1984' by George Orwell
'To Kill a Mockingbird' by Harper Lee
'The Great Gatsby' by F. Scott Fitzgerald

```

Benefits

- **Abstraction:** The Iterator pattern hides the internal structure of the collection, allowing clients to access elements without needing to know how they are stored.
- **Reusability:** The iterator can be reused across different types of collections, providing a uniform way to traverse them.
- **Separation of Concerns:** The responsibility of iteration is separated from the collection itself, making the code cleaner and more modular.

Drawbacks

- **Complexity:** Implementing custom iterators can add complexity to the code, especially for simple collections where built-in iteration methods may suffice.
- **Performance Overhead:** Iterators might introduce a small performance overhead, especially if implemented inefficiently.

The Iterator pattern is particularly useful when you need to provide a standardized way to traverse different types of collections or when the internal structure of the collection should remain hidden from the client. It promotes flexibility and code reusability by decoupling the iteration logic from the collection itself.

### 5. **Mediator**

**Definition:** Defines an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by preventing objects from referring to each other explicitly.

**Use Case:** When you want to manage complex communication between objects or components.

**Example:** Implementing a chatroom where users (components) communicate via a central mediator.

**Definition:**
The Mediator Pattern is a behavioral design pattern that defines an object that encapsulates how a set of objects interact. It promotes loose coupling by preventing objects from referring to each other explicitly, and it allows their interaction to be controlled by a mediator object. This pattern is useful for managing complex communications between objects or components.

**Key Concepts:**

1. **Mediator:** A central object that facilitates communication between multiple objects. It defines a common interface for interacting with the components and manages their interactions.
2. **Colleagues (Components):** The objects that communicate with each other through the mediator. They delegate communication tasks to the mediator instead of directly interacting with each other.

**Structure:**

- **Mediator Interface:** Defines an interface for communication between colleague objects.
- **Concrete Mediator:** Implements the mediator interface and coordinates communication between colleagues.
- **Colleague Classes:** Define a base interface for communication with the mediator and may have methods to send and receive messages through the mediator.

**Example:**

Consider a chatroom application where multiple users (colleagues) can send messages to each other. Instead of users communicating directly, they interact through a central chatroom mediator.

**Mediator Interface:**

```python
from abc import ABC, abstractmethod

class Mediator(ABC):
    @abstractmethod
    def send_message(self, message: str, colleague: 'Colleague') -> None:
        pass

    @abstractmethod
    def add_colleague(self, colleague: 'Colleague') -> None:
        pass

```

**Concrete Mediator:**

```python
class ChatRoomMediator(Mediator):
    def __init__(self):
        self.colleagues = []

    def send_message(self, message: str, colleague: 'Colleague') -> None:
        for c in self.colleagues:
            if c != colleague:
                c.receive(message)

    def add_colleague(self, colleague: 'Colleague') -> None:
        self.colleagues.append(colleague)

```

**Colleague Class:**

```python
class Colleague(ABC):
    def __init__(self, mediator: Mediator):
        self.mediator = mediator

    def send(self, message: str) -> None:
        self.mediator.send_message(message, self)

    @abstractmethod
    def receive(self, message: str) -> None:
        pass

```

**Concrete Colleague:**

```python
class User(Colleague):
    def __init__(self, name: str, mediator: Mediator):
        super().__init__(mediator)
        self.name = name

    def receive(self, message: str) -> None:
        print(f"{self.name} received: {message}")

```

**Client Code:**

```python
def main():
    mediator = ChatRoomMediator()

    user1 = User("Alice", mediator)
    user2 = User("Bob", mediator)
    user3 = User("Charlie", mediator)

    mediator.add_colleague(user1)
    mediator.add_colleague(user2)
    mediator.add_colleague(user3)

    user1.send("Hello everyone!")
    user2.send("Hi Alice!")
    user3.send("Hey!")

if __name__ == "__main__":
    main()

```

**Benefits:**

1. **Loose Coupling:** Mediates interactions between components, reducing direct dependencies between them. Changes to one component do not directly impact others.
2. **Centralized Control:** The mediator centralizes communication logic, making it easier to manage and modify interactions between components.
3. **Simplified Communication:** Simplifies communication by delegating interaction tasks to the mediator, reducing the complexity of interactions between components.
4. **Enhanced Maintainability:** Makes it easier to maintain and extend the system, as components do not need to be aware of each other’s implementation details.
5. **Improved Code Readability:** Provides a clearer structure for communication between components, making the code more readable and understandable.

**When to Use:**

- When you have a set of objects that interact with each other in complex ways, and you want to manage their interactions through a central point.
- When you want to reduce the dependencies between components, making the system more flexible and easier to maintain.
- When you need to provide a centralized mechanism for communication and coordination between different parts of a system.
- When you want to simplify the communication logic and make it easier to understand and modify.

The Mediator Pattern is particularly useful in scenarios where multiple objects need to communicate in a decoupled manner, making the system more modular and maintainable.

### 6. **Memento**

**Definition:** Captures and restores an object’s internal state without violating encapsulation.

**Use Case:** When you need to save and restore the state of an object.

**Example:** Implementing undo functionality in an application by saving snapshots of an object’s state.

The Memento pattern is a behavioral design pattern that allows you to capture and store the current state of an object so that it can be restored to that state later, without exposing the details of the object's implementation. This pattern is often used in scenarios where you want to implement undo/redo functionality.

Example Scenario: Text Editor with Undo Functionality

Imagine a simple text editor where you can type text and undo your last changes. The editor needs to save the current state of the text so it can be restored when the user decides to undo their action.

Python Code Example

```python
class Memento:
    """Memento class stores the state of the Editor."""
    def __init__(self, state):
        self._state = state

    def get_saved_state(self):
        return self._state

class Editor:
    """Editor class that can create and restore mementos."""
    def __init__(self):
        self._content = ""

    def type(self, words):
        self._content += words

    def get_content(self):
        return self._content

    def save(self):
        """Saves the current state inside a memento."""
        return Memento(self._content)

    def restore(self, memento):
        """Restores the state from a memento."""
        self._content = memento.get_saved_state()

class History:
    """Caretaker class responsible for storing mementos."""
    def __init__(self):
        self._mementos = []

    def push(self, memento):
        self._mementos.append(memento)

    def pop(self):
        return self._mementos.pop() if self._mementos else None

# Client code
if __name__ == "__main__":
    editor = Editor()
    history = History()

    # Type some text and save state
    editor.type("This is the first sentence.")
    history.push(editor.save())

    # Type more text and save state
    editor.type(" This is the second sentence.")
    history.push(editor.save())

    # Type even more text
    editor.type(" And this is the third sentence.")
    print("Current Content:", editor.get_content())

    # Undo the last change
    editor.restore(history.pop())
    print("After Undo:", editor.get_content())

    # Undo again
    editor.restore(history.pop())
    print("After Second Undo:", editor.get_content())

```

Explanation

1. **Memento Class:**
    - The `Memento` class is responsible for storing the state of the `Editor` object. It has a method `get_saved_state` to retrieve the stored state.
2. **Editor Class (Originator):**
    - The `Editor` class represents the object whose state needs to be saved and restored. It has methods to type text, save the current state (by creating a `Memento`), and restore a previous state from a `Memento`.
3. **History Class (Caretaker):**
    - The `History` class is responsible for keeping track of the `Memento` objects. It acts as a caretaker that stores the history of the `Editor`'s states. The `push` method adds a new `Memento` to the history, and the `pop` method retrieves the last saved state.
4. **Client Code:**
    - The client code creates an instance of `Editor` and `History`, types some text, and saves the states at different points. It then demonstrates undo functionality by restoring the previous states using the mementos stored in the history.

Output

```
Current Content: This is the first sentence. This is the second sentence. And this is the third sentence.
After Undo: This is the first sentence. This is the second sentence.
After Second Undo: This is the first sentence.

```

Benefits

- **Encapsulation:** The Memento pattern preserves the encapsulation of the originator by not exposing its internal state to other objects.
- **Undo/Redo Support:** It is useful in scenarios where you need to implement undo/redo functionality.
- **State Management:** Allows for easy management and restoration of an object's state at different points in time.

Drawbacks

- **Memory Overhead:** Storing multiple states can consume significant memory, especially if the states are large.
- **Complexity:** Managing mementos and the corresponding history can add complexity to the codebase, especially in applications with many states to track.

The Memento pattern is an excellent choice when you need to allow an object to be restored to a previous state, such as implementing undo functionality in text editors or other applications where state management is crucial. It strikes a balance between state management and encapsulation, ensuring that the internal details of the originator are protected while still providing the ability to save and restore its state.

### 7. **Observer**

**Definition:** Defines a one-to-many dependency between objects, so that when one object changes state, all its dependents are notified and updated automatically.

**Use Case:** When an object’s state changes and you need other objects to be notified and updated.

**Example:** Implementing a notification system where multiple observers receive updates when the subject (observable) changes.

The Observer pattern is a behavioral design pattern that allows an object, known as the subject, to notify other objects, known as observers, about changes in its state. The observers can subscribe to the subject to receive these updates, enabling a one-to-many dependency between objects. This pattern is useful for implementing distributed event handling systems.

Example Scenario: Weather Station

Imagine a weather station system where different display devices (like a mobile app, website widget, and LED screen) need to update their data whenever the weather changes. The weather station acts as the subject, and the display devices are the observers.

Python Code Example

```python
class Subject:
    """The Subject class keeps track of observers and notifies them of any changes."""
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        """Attach an observer to the subject."""
        self._observers.append(observer)

    def detach(self, observer):
        """Detach an observer from the subject."""
        self._observers.remove(observer)

    def notify(self):
        """Notify all observers about an event."""
        for observer in self._observers:
            observer.update(self)

class WeatherStation(Subject):
    """The WeatherStation class represents the subject that monitors weather conditions."""
    def __init__(self):
        super().__init__()
        self._temperature = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        self.notify()  # Notify all observers when the temperature changes

class Observer:
    """The Observer class should be implemented by concrete observers."""
    def update(self, subject):
        pass

class MobileAppDisplay(Observer):
    """A concrete observer that represents a mobile app display."""
    def update(self, subject):
        print(f"Mobile App Display: The current temperature is {subject.temperature}°C.")

class LEDScreenDisplay(Observer):
    """A concrete observer that represents an LED screen display."""
    def update(self, subject):
        print(f"LED Screen Display: The current temperature is {subject.temperature}°C.")

class WebsiteWidgetDisplay(Observer):
    """A concrete observer that represents a website widget."""
    def update(self, subject):
        print(f"Website Widget Display: The current temperature is {subject.temperature}°C.")

# Client code
if __name__ == "__main__":
    # Create the subject (WeatherStation)
    weather_station = WeatherStation()

    # Create observers
    mobile_display = MobileAppDisplay()
    led_display = LEDScreenDisplay()
    website_display = WebsiteWidgetDisplay()

    # Attach observers to the subject
    weather_station.attach(mobile_display)
    weather_station.attach(led_display)
    weather_station.attach(website_display)

    # Change the temperature and notify observers
    weather_station.temperature = 25
    weather_station.temperature = 30

    # Detach an observer and change the temperature again
    weather_station.detach(website_display)
    weather_station.temperature = 28

```

Explanation

1. **Subject Class:**
    - The `Subject` class maintains a list of observers and provides methods to attach, detach, and notify them. The `notify` method calls the `update` method of all registered observers whenever the subject's state changes.
2. **WeatherStation Class (Concrete Subject):**
    - The `WeatherStation` class extends the `Subject` class and represents a concrete subject. It has a `temperature` attribute, and whenever the temperature changes, the `notify` method is called to update all observers.
3. **Observer Class:**
    - The `Observer` class is an interface that defines the `update` method. All concrete observers must implement this method to receive updates from the subject.
4. **Concrete Observers (MobileAppDisplay, LEDScreenDisplay, WebsiteWidgetDisplay):**
    - These classes implement the `Observer` interface and define how they handle updates from the subject. Each observer displays the current temperature in a different way.
5. **Client Code:**
    - The client code creates a `WeatherStation` instance (subject) and several display devices (observers). The observers are attached to the subject, and when the temperature changes, they are notified and display the updated temperature.

Output

```
Mobile App Display: The current temperature is 25°C.
LED Screen Display: The current temperature is 25°C.
Website Widget Display: The current temperature is 25°C.
Mobile App Display: The current temperature is 30°C.
LED Screen Display: The current temperature is 30°C.
Website Widget Display: The current temperature is 30°C.
Mobile App Display: The current temperature is 28°C.
LED Screen Display: The current temperature is 28°C.

```

Benefits

- **Loose Coupling:** The subject and observers are loosely coupled. The subject only knows that it has a list of observers, but it doesn't need to know the specifics of how each observer handles the updates.
- **Flexibility:** New observers can be added or removed at runtime without changing the subject's code.
- **Distributed Updates:** The pattern is ideal for scenarios where multiple objects need to be informed about state changes.

Drawbacks

- **Memory Overhead:** If not managed properly, the list of observers can grow indefinitely, leading to memory overhead.
- **Complexity:** The pattern can introduce complexity when there are many observers or when the interaction between them is not straightforward.
- **Potential Performance Issues:** If there are many observers, the notification process could become time-consuming, especially if the `update` method performs heavy computations.

The Observer pattern is widely used in event-driven programming, such as in GUI frameworks or distributed systems where objects need to respond to changes in other objects. It promotes a clean separation between the core logic (subject) and the components that need to react to changes (observers).

### 8. **State**

**Definition:** Allows an object to alter its behavior when its internal state changes. The object will appear to change its class.

**Use Case:** When an object should change its behavior based on its state.

**Example:** Implementing different behaviors in a game character depending on whether it is idle, walking, or running.

The State pattern is a behavioral design pattern that allows an object to change its behavior when its internal state changes. The pattern delegates state-specific behavior to separate state classes, allowing the context (the object that has state) to switch between these states dynamically.

Example Scenario: Vending Machine

Consider a vending machine that can be in different states: waiting for a coin, waiting for a selection, and dispensing an item. The behavior of the vending machine depends on its current state. For instance, if it's waiting for a coin, pressing a button should have no effect; but if it's waiting for a selection, pressing a button should dispense an item.

Python Code Example

```python
from abc import ABC, abstractmethod

class State(ABC):
    """The base State class defines the interface for all concrete states."""

    @abstractmethod
    def insert_coin(self, machine):
        pass

    @abstractmethod
    def press_button(self, machine):
        pass

class WaitingForCoinState(State):
    """Concrete state where the machine is waiting for a coin to be inserted."""

    def insert_coin(self, machine):
        print("Coin inserted. You can now make a selection.")
        machine.state = machine.waiting_for_selection_state

    def press_button(self, machine):
        print("Please insert a coin first.")

class WaitingForSelectionState(State):
    """Concrete state where the machine is waiting for the user to make a selection."""

    def insert_coin(self, machine):
        print("You've already inserted a coin. Make a selection.")

    def press_button(self, machine):
        print("Selection made. Dispensing item...")
        machine.state = machine.dispensing_state

class DispensingState(State):
    """Concrete state where the machine is dispensing the item."""

    def insert_coin(self, machine):
        print("Please wait, dispensing item...")

    def press_button(self, machine):
        print("Please wait, dispensing item...")

class VendingMachine:
    """The Context class that maintains an instance of a ConcreteState subclass that defines the current state."""

    def __init__(self):
        self.waiting_for_coin_state = WaitingForCoinState()
        self.waiting_for_selection_state = WaitingForSelectionState()
        self.dispensing_state = DispensingState()
        self.state = self.waiting_for_coin_state  # Initial state

    def insert_coin(self):
        self.state.insert_coin(self)

    def press_button(self):
        self.state.press_button(self)

# Client code
if __name__ == "__main__":
    machine = VendingMachine()

    # Test vending machine behavior
    machine.press_button()    # No coin inserted
    machine.insert_coin()     # Insert a coin
    machine.press_button()    # Make a selection
    machine.press_button()    # Trying to press again while dispensing
    machine.insert_coin()     # Try to insert a coin while dispensing
    machine.press_button()    # After dispensing, press again

```

Explanation

1. **State Class:**
    - The `State` class is an abstract base class that defines the interface for different states. It declares methods like `insert_coin` and `press_button` that each concrete state must implement.
2. **Concrete State Classes (WaitingForCoinState, WaitingForSelectionState, DispensingState):**
    - Each concrete state class implements the `State` interface and defines behavior specific to that state. For example, in `WaitingForCoinState`, pressing a button should prompt the user to insert a coin first.
3. **VendingMachine Class (Context):**
    - The `VendingMachine` class acts as the context that maintains an instance of a concrete state class that defines the current state. It provides methods like `insert_coin` and `press_button` that delegate behavior to the current state. The state can change dynamically as the machine operates.
4. **State Transitions:**
    - State transitions are managed within the concrete state classes. For instance, in `WaitingForCoinState`, after a coin is inserted, the state of the machine is changed to `waiting_for_selection_state`.
5. **Client Code:**
    - The client interacts with the `VendingMachine` object by calling its methods (`insert_coin` and `press_button`). The machine’s behavior changes dynamically depending on its current state.

Output

```
Please insert a coin first.
Coin inserted. You can now make a selection.
Selection made. Dispensing item...
Please wait, dispensing item...
Please wait, dispensing item...
Please wait, dispensing item...

```

Benefits

- **Encapsulation of State-Specific Behavior:** The pattern encapsulates the behavior associated with a particular state, making the code easier to understand and modify.
- **Open/Closed Principle:** New states and transitions can be added without modifying existing code, adhering to the Open/Closed Principle.
- **Simplified Conditionals:** Instead of having complex conditional logic (e.g., `if-else` or `switch-case`) spread across the class, the behavior for each state is neatly encapsulated within its own class.

Drawbacks

- **Increased Number of Classes:** The pattern increases the number of classes in the system, which can make the codebase larger and potentially harder to manage.
- **Tight Coupling Between States and Context:** The context and state classes are closely coupled, as the context needs to know about all possible states. However, this coupling is necessary to achieve the dynamic behavior.

Use Cases

- **Finite State Machines:** The State pattern is ideal for implementing finite state machines where an object needs to transition between a set number of states.
- **UI Components:** User interface elements that can be in different states (e.g., buttons, form fields) can benefit from the State pattern to handle different behaviors based on the state.
- **Game Development:** In game development, entities like characters or game objects that have different behaviors depending on their state (e.g., walking, jumping, attacking) can use the State pattern to manage state-specific logic.

### 9. **Strategy**

**Definition:** Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

**Use Case:** When you have multiple algorithms for a specific task and want to choose one at runtime.

**Example:** Implementing different sorting algorithms that can be swapped depending on the context.

The Strategy pattern is a behavioral design pattern that enables selecting an algorithm's behavior at runtime. Instead of implementing a single algorithm directly, code receives run-time instructions as to which in a family of algorithms to use. This pattern allows the client to choose which algorithm to use for a specific context.

Example Scenario: Payment Processing

Consider a payment processing system where different payment methods (e.g., credit card, PayPal, bank transfer) are available. The payment processing system should be able to handle multiple payment methods, and the user should be able to choose the method at runtime.

Python Code Example

```python
from abc import ABC, abstractmethod

# Strategy Interface
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

# Concrete Strategies
class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} using Credit Card.")

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} using PayPal.")

class BankTransferPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} using Bank Transfer.")

# Context
class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: PaymentStrategy):
        self.strategy = strategy

    def process_payment(self, amount):
        self.strategy.pay(amount)

# Client code
if __name__ == "__main__":
    # Choose payment strategy at runtime
    payment_processor = PaymentProcessor(CreditCardPayment())
    payment_processor.process_payment(100)

    payment_processor.set_strategy(PayPalPayment())
    payment_processor.process_payment(150)

    payment_processor.set_strategy(BankTransferPayment())
    payment_processor.process_payment(200)

```

Explanation

1. **Strategy Interface (PaymentStrategy):**
    - The `PaymentStrategy` class is an abstract base class that defines a common interface for all payment strategies. The `pay` method must be implemented by all concrete strategies.
2. **Concrete Strategy Classes (CreditCardPayment, PayPalPayment, BankTransferPayment):**
    - These classes implement the `PaymentStrategy` interface and provide concrete implementations for the `pay` method. Each class represents a different payment method.
3. **Context (PaymentProcessor):**
    - The `PaymentProcessor` class uses a `PaymentStrategy` object to execute the payment. The strategy can be set at runtime, allowing the client to change the payment method without altering the `PaymentProcessor` class itself.
4. **Client Code:**
    - The client creates a `PaymentProcessor` object and initially sets it to use the `CreditCardPayment` strategy. Later, the payment strategy is changed to `PayPalPayment` and `BankTransferPayment` by calling the `set_strategy` method. This allows for flexibility in choosing different payment methods at runtime.

Output

```
Paid 100 using Credit Card.
Paid 150 using PayPal.
Paid 200 using Bank Transfer.

```

Benefits

- **Flexibility:** The Strategy pattern provides flexibility to choose and change the algorithm at runtime, making the system more adaptable.
- **Single Responsibility Principle:** By separating different algorithms into their own classes, each class has a single responsibility, adhering to the Single Responsibility Principle.
- **Open/Closed Principle:** New strategies (algorithms) can be added without modifying existing code, adhering to the Open/Closed Principle.

Drawbacks

- **Increased Number of Classes:** The Strategy pattern can result in an increased number of classes in the system, which can make the codebase larger and potentially harder to manage.
- **Overhead in Selecting Strategy:** If the selection of the strategy involves complex decision-making, there can be overhead in determining which strategy to use at runtime.

Use Cases

- **Payment Systems:** The Strategy pattern is often used in payment systems where different payment methods are supported.
- **Sorting Algorithms:** If an application needs to support different sorting algorithms (e.g., quicksort, mergesort, bubblesort), the Strategy pattern can be used to select the sorting algorithm at runtime.
- **File Compression:** The Strategy pattern can be used in file compression systems where different compression algorithms (e.g., zip, gzip, rar) are supported, allowing the user to choose the desired algorithm at runtime.
- **Data Validation:** In applications that require different validation algorithms depending on the context, the Strategy pattern allows for flexible selection of the appropriate validation strategy.

### 10. **Template Method**

**Definition:** Defines the skeleton of an algorithm in a method, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm’s structure.

**Use Case:** When you have a fixed sequence of steps for an algorithm and want to allow customization of some steps.

**Example:** Implementing a data processing pipeline where certain steps (e.g., data loading, processing, and saving) are fixed but can be customized.

Template Method Pattern

The Template Method pattern is a behavioral design pattern that defines the skeleton of an algorithm in a method, deferring some steps to subclasses. The Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.

Example Scenario: Data Processing Pipeline

Consider a data processing system where data is fetched from various sources, processed, and then saved. The overall structure of the process is the same, but the specifics of fetching, processing, and saving can vary depending on the source and format of the data.

Python Code Example

```python
from abc import ABC, abstractmethod

# Abstract class that defines the template method
class DataProcessor(ABC):
    # Template method
    def process_data(self):
        self.fetch_data()
        self.process_data_logic()
        self.save_data()

    # Concrete methods (implemented in the base class)
    def fetch_data(self):
        print("Fetching data...")

    # Abstract methods (steps that must be implemented by subclasses)
    @abstractmethod
    def process_data_logic(self):
        pass

    @abstractmethod
    def save_data(self):
        pass

# Concrete class 1
class CSVDataProcessor(DataProcessor):
    def process_data_logic(self):
        print("Processing data from CSV...")

    def save_data(self):
        print("Saving processed CSV data...")

# Concrete class 2
class APIDataProcessor(DataProcessor):
    def process_data_logic(self):
        print("Processing data from API...")

    def save_data(self):
        print("Saving processed API data...")

# Client code
if __name__ == "__main__":
    csv_processor = CSVDataProcessor()
    csv_processor.process_data()

    api_processor = APIDataProcessor()
    api_processor.process_data()

```

Explanation

1. **Template Method (process_data):**
    - The `process_data` method in the `DataProcessor` class defines the overall structure of the algorithm. This method calls several steps, some of which are implemented in the base class (`fetch_data`), and others that are left abstract (`process_data_logic` and `save_data`) and must be implemented by subclasses.
2. **Abstract Class (DataProcessor):**
    - `DataProcessor` is an abstract class that provides the template method (`process_data`) and defines the abstract methods (`process_data_logic` and `save_data`) that subclasses must implement.
3. **Concrete Classes (CSVDataProcessor, APIDataProcessor):**
    - These classes extend the `DataProcessor` class and provide specific implementations for the abstract methods. For example, `CSVDataProcessor` processes CSV data, while `APIDataProcessor` processes data from an API.
4. **Client Code:**
    - The client creates instances of the concrete processors (`CSVDataProcessor` and `APIDataProcessor`) and calls the `process_data` method. The template method ensures that the data processing follows the same steps, but the details of each step are determined by the specific subclass.

Output

```
Fetching data...
Processing data from CSV...
Saving processed CSV data...
Fetching data...
Processing data from API...
Saving processed API data...

```

### Benefits

- **Code Reuse:** The Template Method pattern promotes code reuse by allowing the common steps of an algorithm to be defined in a single place (the abstract class) and specific steps to be implemented in subclasses.
- **Flexibility:** Subclasses can vary the implementation of specific steps without altering the overall structure of the algorithm.
- **Enforces a Structure:** The pattern enforces a certain structure or workflow, ensuring that steps are executed in a specific order.

Drawbacks

- **Inheritance Overuse:** The Template Method pattern relies on inheritance, which can lead to a rigid class hierarchy and reduced flexibility if not managed carefully.
- **Increased Complexity:** The pattern can introduce complexity, especially if there are many steps and subclasses, making the code harder to understand and maintain.

Use Cases

- **Data Processing Pipelines:** When you have a sequence of steps for processing data, and some steps may vary depending on the data source or format, the Template Method pattern is an ideal choice.
- **UI Frameworks:** In user interface frameworks, the Template Method pattern is often used to define the general structure of rendering UI components, allowing specific components to customize parts of the rendering process.
- **Game Development:** In game development, the Template Method pattern can be used to define the structure of a game loop, where specific games can customize the behavior for initialization, input processing, updating, and rendering.
- **Report Generation:** The pattern can be used in report generation systems where the overall structure of report generation is fixed, but different types of reports require different data processing and formatting steps.

The Template Method pattern is particularly useful in situations where you want to enforce a specific workflow or algorithm structure but allow parts of it to be customized or extended by subclasses.

### 11. **Visitor**

**Definition:** Defines a new operation to a set of objects without changing the objects themselves. The Visitor pattern separates algorithms from the objects on which they operate.

**Use Case:** When you need to perform operations on a set of objects that may have different types.

**Example:** Implementing operations like rendering or serialization on elements of a complex object structure (e.g., file system).

The Visitor pattern is a behavioral design pattern that allows you to add new operations to existing class hierarchies without modifying the existing code. Essentially, it separates the algorithm from the object structure on which it operates.

**Key Components:**

- **Element Interface:** Defines an interface for elements in the object structure.
- **Concrete Elements:** Implement the Element interface and represent different types of objects in the structure.
- **Visitor Interface:** Defines an interface for visitor operations.
- **Concrete Visitors:** Implement the Visitor interface and define specific operations to be performed on the elements.

**How it Works:**

1. An element in the object structure accepts a visitor object.
2. The visitor object performs the appropriate operation based on the element type.

**Use Cases**

- **Adding new operations to existing object structures:** When you need to perform different actions on various object types, without modifying the base classes.
- **AST (Abstract Syntax Tree) traversal:** Analyzing and manipulating code structures.
- **Shape drawing:** Different ways to draw shapes (e.g., SVG, bitmap) without changing the shape classes.
- **Report generation:** Generating different types of reports from the same data structure.

**Benefits**

- **Open/Closed Principle:** Allows adding new operations without modifying existing code.
- **Improved code organization:** Separates algorithms from data structures.
- **Flexibility:** Enables easy addition of new operations.
- **Potential for performance optimizations:** Specific optimizations can be implemented within visitor classes.

**Drawbacks**

- **Increased complexity:** The pattern introduces additional classes and interfaces, which might complicate the codebase.
- **Potential for tight coupling:** If not carefully designed, the Visitor and Element classes can become tightly coupled.
- **Acceptance method:** The `accept` method in elements can lead to boilerplate code.

**Python**

`from abc import ABC, abstractmethod

class Element(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass

class ConcreteElementA(Element):
    def accept(self, visitor):
        visitor.visit_concrete_element_a(self)   class ConcreteElementB(Element):
    def accept(self, visitor):
        visitor.visit_concrete_element_b(self)

class Visitor(ABC):
    @abstractmethod
    def visit_concrete_element_a(self, element):
        pass

    @abstractmethod
    def visit_concrete_element_b(self, element):
        pass

class ConcreteVisitor(Visitor):   def visit_concrete_element_a(self, element):
        print("Visiting ConcreteElementA")
        # Perform specific operation on element

    def visit_concrete_element_b(self, element):
        print("Visiting ConcreteElementB")
        # Perform specific operation on element

# Usage
element_a = ConcreteElementA()
element_b = ConcreteElementB()
visitor = ConcreteVisitor()

element_a.accept(visitor)
element_b.accept(visitor)`

**Conclusion**

The Visitor pattern is a powerful tool for adding new behaviors to existing object structures in Python. While it introduces some complexity, it can significantly improve code maintainability and flexibility in certain scenarios. Carefully consider the trade-offs before applying it to your project.

### 12. **Null Object**

**Definition:** Provides an object representing a default or neutral behavior that can be used in place of null references. This avoids the need for null checks.

**Use Case:** When you want to avoid null checks and handle absent or default behavior in a clean way.

**Example:** Implementing a null object for a logging interface that performs no operation, avoiding null checks in client code.

These behavioral patterns help manage and simplify object interactions, ensuring that systems are more maintainable, scalable, and flexible.

The Null Object pattern is a design pattern that provides a default implementation for an object. This default implementation does nothing or returns a neutral value. The primary purpose is to avoid null pointer exceptions and simplify code by providing a consistent interface for both existing and non-existing objects.

**Use Cases**

- **Handling optional objects:** When dealing with objects that might be null or absent, a null object can provide a default behavior without explicit null checks.
- **Simplifying conditional logic:** By replacing null checks with calls to a null object, you can reduce the complexity of your code.
- **Placeholder objects:** You can use a null object as a placeholder for objects that haven't been created yet or are not available.

**Benefits**

- **Improved code readability:** By eliminating null checks, code becomes more concise and easier to understand.
- **Error prevention:** Prevents null pointer exceptions and related runtime errors.
- **Default behavior:** Provides a consistent default behavior for objects that might be absent.
- **Simplified code:** Reduces the need for conditional logic to handle null values.

**Drawbacks**

- **Increased complexity:** Introducing a new object type can increase the complexity of the codebase.
- **Overuse:** Using the null object pattern for every potential null value can lead to over-engineering.
- **Potential for misuse:** The null object might hide underlying issues if not used carefully.

**Python**

`class AbstractLogger:
    def log(self, message):
        pass

class ConsoleLogger(AbstractLogger):
    def log(self, message):
        print(message)

class NullLogger(AbstractLogger):
    def log(self, message):
        pass

# Usage
logger = ConsoleLogger()  # Or logger = NullLogger()
logger.log("This is a log message")`

In this example, `NullLogger` serves as a null object that does nothing when the `log` method is called.

**Conclusion**

The Null Object pattern is a valuable tool for handling potential null values in your code. By providing a default implementation, it can improve code readability, reduce errors, and simplify logic. However, it's essential to use it judiciously to avoid over-complicating your codebase.
