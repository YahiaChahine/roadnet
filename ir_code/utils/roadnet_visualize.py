import json
import matplotlib.pyplot as plt
import sys, os


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)
    with open('./data/xuancheng/roadnet_xuancheng241007.json', 'r') as file:
        roadnet = json.load(file)
    
    intersections, roads = roadnet['intersections'], roadnet['roads']

    plt.Figure(figsize=(20, 20))
    # for inter in intersections:
    #     plt.scatter(inter['point']['x'], inter['point']['y'], c='k', s=0.5)
        # plt.annotate(inter['id'].replace('intersection_', ''), xy=(inter['point']['x'] + 20, inter['point']['y'] + 2), fontsize=5)
    for road in roads:
        plt.plot([road['points'][0]['x'], road['points'][1]['x'],road['points'][2]['x'],road['points'][3]['x']], [road['points'][0]['y'], road['points'][1]['y'],road['points'][2]['y'],road['points'][3]['y']], c='k', linewidth=0.5)

    plt.savefig('./utils/validate/show.png', dpi=500)
    # plt.show()