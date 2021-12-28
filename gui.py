from os import stat
import tkinter as tk
from tkinter import ttk

HEIGHT = 500
WIDTH = 300

class GUI():
    
    def __init__(self, title: str, geometry: str):
        super().__init__()
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry(geometry)
        self.window.configure(background = "#34A2FE")
        self.tabControl = ttk.Notebook(self.window)
        self.tab_one = ttk.Frame(self.tabControl)
        
        self.tab_one.pack()
        self.tab_two = ttk.Frame(self.tabControl)
        self.tab_two.pack()
        self.build_tabControl()
        self.tabControl.pack(expand=1, fill="both")
        self.window.update_idletasks()
        self.window.update()
      
        
    
    def build_tabControl(self):
    
        self.tabControl.add(self.tab_one, text='Live Test')
        self.tabControl.add(self.tab_two, text='Test Results')
        
        self.console = tk.Text(self.tab_one, width=800, height=600)
        self.console.pack()
        self.test_console = tk.Text(self.tab_two, width=800, height=600)
        self.test_console.pack()
        
        
    
    @staticmethod
    def print_to_tab(message:str, textBox):
        textBox.insert(tk.END, message)
        
        
        
    
    
    

        
    
   
        
    
    


    