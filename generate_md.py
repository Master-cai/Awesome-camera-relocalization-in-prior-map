import os
import json
from operator import itemgetter
import sys

def generate_one_piece(info):
    """
    generate one piece of markdown with some badges

    parameters
    ----------
    info: dict
        a dict include 'title', 'year', 'pub', 'url', 'code'(not necessary)

    return
    ------
    str
        a piece of markdown
    """
    if 'code' in info.keys() and info['code'] != "":
        return f"![](https://img.shields.io/badge/year-{info['year']}-g)![](https://img.shields.io/badge/pub-{info['pub']}-orange)[{info['title']}]({info['url']}) |[code]({info['code']})![](https://img.shields.io/github/stars/{info['code'].split('/')[-2]}/{info['code'].split('/')[-1]}?style=social)"
        # return f"![](https://img.shields.io/badge/year-{info['year']}-g)![](https://img.shields.io/badge/pub-{info['pub']}-orange)[{info['title']}]({info['url']}) |[link]({info['code']})"
    elif 'page' in info.keys() and info['page'] != "":
        return f"![](https://img.shields.io/badge/year-{info['year']}-g)![](https://img.shields.io/badge/pub-{info['pub']}-orange)[{info['name']}]({info['url']}) |[page]({info['page']})"
    else:
        return f"![](https://img.shields.io/badge/year-{info['year']}-g)![](https://img.shields.io/badge/pub-{info['pub']}-orange)[{info['title']}]({info['url']})"
    

def generate_all_pieces(data):
    """
    generate all pieces of paper markdown

    parameters
    ----------
    data: list
        a list of dict include 'title', 'year', 'pub', 'url', 'code'(not necessary)

    return
    ------
    str
        all pieces of markdown
    """
    # sort by year, then conference
    data = sorted(data, key=itemgetter('year', 'pub'))

    return "\n\n".join([generate_one_piece(m) for m in data])

def generate_markdown(folder_path, output_file):
    """
    generate markdown file according to the folder structure and its contents

    parameters
    ----------
    folder_path: str
        the path of the folder
    output_file: str
        the path of the output markdown file
    """
    with open(output_file, 'w') as file:
        markdown_content, table_of_contents = generate_markdown_recursive(folder_path, 1)
        file.write(table_of_contents)
        file.write('\n')
        file.write(markdown_content)

def generate_markdown_recursive(folder_path, level):
    """
    generate markdown for current level

    parameters
    ----------
    folder_path: str
        the path of the folder
    level: int
        the level of the folder

    return
    ------
    str
        the markdown content of current level
    """
    folder_name = os.path.basename(folder_path)
    markdown_content = f"{'#' * level} {folder_name}\n\n"
    table_of_contents = f"{'  ' * level}- [{folder_name}](#{folder_name.lower().replace(' ', '-')})\n"


    if os.path.isfile(os.path.join(folder_path, 'intro.txt')):
        with open(os.path.join(folder_path, 'intro.txt'), 'r') as intro_file:
            markdown_content += intro_file.read() + '\n'
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        if os.path.isdir(item_path):
            item_content, item_toc = generate_markdown_recursive(item_path, level + 1)
                        
            markdown_content += item_content
            table_of_contents += item_toc
        elif os.path.isfile(item_path) and item.endswith('.json'):
            with open(item_path, 'r') as f:
                data = json.load(f)
                json_title = data['title']
                markdown_content += f"{'#' * (level + 1)} {json_title}\n\n"
                if 'intro' in data.keys():
                    markdown_content += data['intro'] + '\n\n'
                markdown_content += generate_all_pieces(data['papers'])
            
            markdown_content += '\n'
            
            table_of_contents += f"{'  ' * (level+1)}- [{json_title}](#{json_title.lower().replace(' ', '-')})\n"
    
    return markdown_content, table_of_contents

folder_path = 'json_data'
output_file = 'output.md'
generate_markdown(folder_path, output_file)