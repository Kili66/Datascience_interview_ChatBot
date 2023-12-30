from tkinter import *
from chat import get_response, bot_name

BG_Gray="#ABB2B9"
BG_Color="#17202A"
Text_color="#EAECEE"

Font= "Helvetica 14"
Font_bold= "Helvetica 13 bold"

class ChatApplication:
    def __init__(self):
        self.window= Tk()
        self._setup_main_window()
    def run(self):
        self.window.mainloop()
    def _setup_main_window(self):
        self.window.title("Chat Bot")
        self.window.resizable(width=True, height=True)
        self.window.configure(width=470, height=550, bg=BG_Color)

        # Head label
        head_label= Label(self.window, bg=BG_Color, fg=Text_color, text="Welcome!", font=Font_bold, pady=10)
        head_label.place(relwidth=1)
        
        # tiny divider
        line= Label(self.window, width=450, bg=BG_Gray)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        # Text widget # where text is displayed
        self.text_widget= Text(self.window, width=20, height=2, bg=BG_Color, fg=Text_color, font=Font, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        
        #scroll bar
        scrollbar= Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)
        
        # button label
        bottom_label= Label(self.window, bg=BG_Gray, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        # message entry box
        
        self.msg_entry= Entry(bottom_label, bg="#2C3E5a", fg=Text_color, font=Font)
        self.msg_entry.place(relwidth=1, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # send button
        send_button= Button(bottom_label, text="Send", font=Font_bold, width=20, bg=BG_Gray, command= lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
    def _on_enter_pressed(self, event):
        msg= self.msg_entry.get()
        self._insert_message(msg, "You")
    def _insert_message(self, msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0, END) 
        msg1=f"{sender}: {msg}\n\n"   
        #enable
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)  
        self.text_widget.configure(state=DISABLED) 
        
        # get response from chat
        msg2=f"{bot_name}: {get_response(msg)}\n\n"   
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)  
        self.text_widget.configure(state=DISABLED) 
        self.text_widget.see(END)
if __name__=="__main__":
    app= ChatApplication()  
    app.run()