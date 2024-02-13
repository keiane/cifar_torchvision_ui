from imports import *

def normalize(img):
    min_im = np.min(img)
    np_img = img - min_im
    max_im = np.max(np_img)
    np_img /= max_im
    return np_img

def imshow(img, fig_name = "test_input.png"):
    try:
        img = img.clone().detach().cpu().numpy()
    except:
        print('img already numpy')

    plt.imshow(normalize(np.transpose(img, (1, 2, 0))))
    plt.savefig(fig_name)
    print(f'Figure saved as {fig_name}')
    return fig_name

def class_names(class_num, class_list): # converts the raw number label to text
    if class_num == 0:
        return(class_list[0])
    elif class_num == 1:
        return(class_list[1])
    elif class_num == 2:
        return(class_list[2])
    elif class_num == 3:
        return(class_list[3])
    elif class_num == 4:
        return(class_list[4])
    elif class_num == 5:
        return(class_list[5])
    elif class_num == 6:
        return(class_list[6])
    elif class_num == 7:
        return(class_list[7])
    elif class_num == 8:
        return(class_list[8])
    elif class_num == 9:
        return(class_list[9])
    
def togglepicsettings(choice):
    yes=gr.Gallery(visible=True)
    no=gr.Gallery(visible=False)
    if choice == "Yes":
        return yes,no
    else:
        return no,yes

def settings(choice):
    if choice == "Advanced":
        advanced = [
        gr.Slider(visible=True),
        gr.Slider(visible=True),
        gr.Slider(visible=True),
        gr.Dropdown(visible=True),
        gr.Dropdown(visible=True),
        gr.Radio(visible=True)
        ]
        return advanced
    else:
        basic = [
        gr.Slider(visible=False),
        gr.Slider(visible=False),
        gr.Slider(visible=False),
        gr.Dropdown(visible=False),
        gr.Dropdown(visible=False),
        gr.Radio(visible=False)
        ]
        return basic

def attacks(choice):
    if choice == "Yes":
        yes = [
            gr.Markdown(visible=True),
            gr.Radio(visible=True),
            gr.Radio(visible=True)
        ]
        return yes
    if choice == "No":
        no = [
            gr.Markdown(visible=False),
            gr.Radio(visible=False),
            gr.Radio(visible=False)
        ]
        return no
    
def gaussian(choice):
    if choice == "Yes":
        yes = [
            gr.Slider(visible=True),
            gr.Gallery(visible=True),
        ]
        return yes
    else:
        no = [
            gr.Slider(visible=False),
            gr.Gallery(visible=False),
        ]
        return no
def adversarial(choice):
    if choice == "Yes":
        yes = gr.Gallery(visible=True)
        return yes
    else:
        no = gr.Gallery(visible=False)

def input_protection(drop_type, username, epochs_sldr, train_sldr, test_sldr):
    if not drop_type:
        gr.Warning("Please select a model from the dropdown.")
        return
    if not username:
        gr.Warning("Please enter a WandB username.")
        return
    if(epochs_sldr % 1 != 0):
        gr.Warning("Number of epochs must be an integer.")
        return
    if(train_sldr % 1 != 0):
        gr.Warning("Training batch size must be an integer.")
        return
    if(test_sldr % 1 != 0):
        gr.Warning("Testing batch size must be an integer.")
        return

def documentation_import():
    markdown_file_path = 'documentation.md'
    with open(markdown_file_path, 'r') as file:
        markdown_content = file.read()
    return markdown_content

creators_array = ["henry", "luke", "keiane", "evelyn", "ethan", "matt"]
def creators_import():
    all_content = {}
    for creator in creators_array:
        markdown_creator_path = os.path.join('creators', creator, f'markdown_{creator}.md')
        with open(markdown_creator_path, 'r') as file:
            markdown_content = file.read()
            all_content[creator] = markdown_content
    
    return all_content