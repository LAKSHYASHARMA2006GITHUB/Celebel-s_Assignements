class Node:
   # Represents a node in the singly linked list.
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    #Represents the singly linked list.
    def __init__(self):
        self.head = None

    def add_node(self, data):
      #  Adds a node to the end of the list.
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def print_list(self):
        #Prints all elements in the list.
        current = self.head
        if current is None:
            print("List is empty.")
            return
        output = ""
        while current:
            output += str(current.data)
            if current.next:
                output += " -> "
            current = current.next
        print(output)

    def delete_nth_node(self, n):
        #Deletes the nth node 
        if self.head is None:
            raise Exception("Can't delete from an empty list.")

        if n <= 0:
            raise IndexError("Index must be 1 or greater.")
        
        if n == 1:
            # Delete head
            self.head = self.head.next
            return

        current = self.head
        prev = None
        count = 1
        
        while current and count < n:
            prev = current
            current = current.next
            count += 1
        
        if current is None:
            raise IndexError("Index out of range.")
        
        prev.next = current.next

# === Test the LinkedList implementation ===
if __name__ == "__main__":
    # Create a linked list
    LL = LinkedList()
    
    # Add nodes
    LL.add_node(10)
    LL.add_node(20)
    LL.add_node(30)
    LL.add_node(40)
    
    print("Initial list:")
    LL.print_list()

    # Delete the 2nd node
    try:
        LL.delete_nth_node(2)
        print("After deleting 2nd node:")
        LL.print_list()
    except Exception as e:
        print(f"Error: {e}")
    
    # Delete the 1st node
    try:
        LL.delete_nth_node(1)
        print("After deleting 1st node:")
        LL.print_list()
    except Exception as e:
        print(f"Error: {e}")
    
    # Attempt to delete a node out of range
    try:
        LL.delete_nth_node(10)
    except Exception as e:
        print(f"Error: {e}")
    
    # Delete remaining nodes
    try:
        LL.delete_nth_node(1)
        LL.delete_nth_node(1)
        # Try deleting from empty list
        LL.delete_nth_node(1)
    except Exception as e:
        print(f"Error: {e}")

    # Additional test: add and print after clearing the list
    try:
        LL.add_node(50)
        LL.add_node(60)
        print("After adding nodes to cleared list:")
        LL.print_list()
    except Exception as e:
        print(f"Error: {e}")
