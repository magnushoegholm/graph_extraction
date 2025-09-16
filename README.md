# Graph Extraction from STEP Files

This project helps you extract kinematic relationships between components in STEP files, turning CAD assemblies into graph data. The code is inspired by tpaviot's pythonocc-core, especially the DataExchange utilities, but here the focus is on finding how parts are connected and interact. Yes, it is spaghetti code :-D But with some AI it should be possible to get an idea of what is going on. 
The starting point is file.py which calls all the other scripts.

## How it works

- You give it a STEP file (a common format for 3D CAD models).
- The code loads the file and finds all the solid components inside.
- For each component, it figures out its bounding box and center. The 'true center' was just a quick fix because the step reader that can get the component name could not also get the correct center of the components (it would just position them all in the origin, making it useless for extraction of kinematic relationships)
- It builds a graph where each node is a component, and edges represent kinematic constraints (how parts are related or move together).
- The result is a dictionary with all the components, their properties, and the edges between them

## Main files

- `file.py`: The main script. Loads STEP files, extracts components, computes centers, and builds the graph.
- `graph_object.py`: Defines the graph structure and methods for adding components and edges.
- `true_center_tool.py`: Calculates the 'true center' of each component for more precise graph building.
- `string_clustering.py`: (Optional) Used for clustering or grouping components based on names or other properties.

## Requirements

This project uses [pythonocc-core](https://github.com/tpaviot/pythonocc-core) for working with STEP files. You can install it via pip:

```bash
pip install pythonocc-core
```

For more details and platform-specific instructions, see the official installation guide: https://github.com/tpaviot/pythonocc-core#installation

## Usage

Just run `file.py` with a STEP file path. It will process the file, extract the graph, and save the results to `graph_data.txt`.

```bash
python file.py
```


---
Inspired by [tpaviot/pythonocc-core](https://github.com/tpaviot/pythonocc-core).