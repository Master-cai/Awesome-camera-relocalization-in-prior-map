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

    return "\n".join([generate_one_piece(m) for m in data])


if __name__ == "__main__":

    if len(sys.argv) == 2:
        file_name = sys.argv[1]
    else:
        file_name = "Image appearance normalization.json"
    with open(f'json_data/{file_name}', 'r') as f:
        data = json.load(f)
    
    all_pieces = generate_all_pieces(data[1])
    print(all_pieces)
