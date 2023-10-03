import os
import random
import pandas as pd
import numpy as np
import imageio.v2 as imageio

from graphviz import Digraph
from IPython.display import Image as ImageDisplay
from PIL import ImageDraw
from PIL import Image
from typing import List, Any

class MarkovChain:
    """
    Represents a Markov chain and provides methods for working with Markov chains.

    Attributes:
        region_list (list): A list of strings representing the regions in the Markov chain.
        remove_self_flight (bool): A flag indicating whether to remove self-flights in the Markov chain.

    Methods:
        baseline_data(): Generates a baseline dataframe based on the region list.
        clear_all_generated_files(): Clears all generated files in the current directory.
        prob_ending_region_after_n_step(matrix, 
                                        initial_region,
                                        final_region,
                                        n_steps): Calculates the probability of ending in a 
                                                  specific region after taking a certain 
                                                  number of steps in a matrix.
        plot_base(): Generates the base plot for visualization.
        block_nodes(matrix,
                    node_1=None,
                    node_2=None): Generates an adjacency matrix based 
                                  on a given DataFrame, with the option to block nodes.
        plot_travel_simulation(plot_mode='base'): Plots the travel simulation.
        preprocess_data(matrix): Preprocesses a given adjacency matrix DataFrame to be an edge-list DataFrame.
        matrix_power(matrix, power): Calculates the power of a matrix.
    """
    def __init__(self,
                 region_list: List[str],
                 remove_self_flight: bool=False) -> None:
        """
        Initializes a new instance of the class.

        Parameters:
            region_list (List[str]): A list of regions.
            remove_self_flight (bool, optional): Indicates whether to remove self flights. Defaults to False.

        Returns:
            None
        """
        self.remove_self_flight = remove_self_flight
        self.region_list = region_list
        return None
        
    def generate_list(self,
                      len_list: int,
                      zero_index_list: List[int]) -> List[float]:
        """Generates a list of probability numbers between 0 and 1,
        the sum of it must be 1.

        Parameters:
            - len_list: length of list
            - zero_index_list: zero_index_list will set the i-th item in the list to be 0
        Example:
            generate_list(4, [0,2])
            result = [0.0, 0.5, 0, 0.5]

        Returns:
            A list of len_list floats.
        """
        list1: List[float] = []
        while sum(list1) < 1.0:
            for i in range(len_list):
                if i in zero_index_list:
                    list1.append(0)
                else:
                    list1.append(random.random())
            sum1 = sum(list1)
            for i in range(len_list):
                if list1[i] != 0:
                    list1[i] = round(list1[i] / sum1, 2)  # round into 2 decimal
            if sum(list1) != 1.0:
                list1 = []
        return list1

    def baseline_data(self,
                      region_list: List[str],
                      remove_self_flight: bool = False) -> pd.DataFrame:
        """
        Generate a baseline dataframe based on a list of regions.

        Parameters:
            - region_list: A list of strings representing the regions.
            - remove_self_flight: A boolean indicating whether to remove self-flights 
                                  from the generated data. Default is False.

        Returns:
            - df: A pandas DataFrame containing the generated baseline data rounded to 2 decimal places.
        """
        data = {}
        for i in region_list:
            if remove_self_flight:
                data[i] = self.generate_list(len(region_list), [region_list.index(i)])
            else:
                data[i] = self.generate_list(len(region_list), [])
        df = pd.DataFrame.from_dict(data, orient='index', columns=region_list)
        return df.round(2)


    def block_nodes(self,
                    df: pd.DataFrame, 
                    node_1: Any = None, 
                    node_2: Any = None, 
                    remove_self_flight: bool = False) -> pd.DataFrame:
        """
        Generates an adjacency matrix based on the given DataFrame, node_1, and node_2.
        The edge value between node_1 and node_2 will be set to be 0.

        Parameters:
            df (pd.DataFrame): The DataFrame to generate the adjacency matrix from.
            node_1 (Any, optional): The first node to consider. Defaults to None.
            node_2 (Any, optional): The second node to consider. Defaults to None.
            remove_self_flight (bool, optional): Whether to remove self flight. Defaults to False.

        Returns:
            pd.DataFrame: The adjacency matrix rounded to two decimal places.
        """
        region_list = list(df.columns)
        if node_1 is None and node_2 is None:
            return df.round(2)
        else:
            adjacency_matrix = np.zeros((len(region_list), len(region_list)))
            for i, region in enumerate(region_list):
                if remove_self_flight:
                    if region == node_1:
                        adjacency_matrix[i, :] = self.generate_list(len(region_list), [i, region_list.index(node_2)])
                    elif region == node_2:
                        adjacency_matrix[i, :] = self.generate_list(len(region_list), [i, region_list.index(node_1)])
                    else:
                        adjacency_matrix[i, :] = self.generate_list(len(region_list), [i])
                else:
                    if region == node_1:
                        adjacency_matrix[i, :] = self.generate_list(len(region_list), [region_list.index(node_2)])
                    elif region == node_2:
                        adjacency_matrix[i, :] = self.generate_list(len(region_list), [region_list.index(node_1)])    
                    else:
                        adjacency_matrix[i, :] = self.generate_list(len(region_list), [])
            df = pd.DataFrame(adjacency_matrix, index=region_list, columns=region_list)
            return df.round(2)

    def preprocess_data(self,
                        df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the given adjacency matrix dataframe
        to be edge-list dataframe.

        Parameters:
            df (pd.DataFrame): The DataFrame to be preprocessed.

        Returns:
            pd.DataFrame: Edge-list dataframe
        """
        return df.rename_axis('source')\
                .reset_index()\
                .melt('source', value_name='weight', var_name='target')\
                .reset_index(drop=True)

    # Input data from dataframe into graph network
    def create_graph_network(self,
                             df: pd.DataFrame,
                             colored: str,
                             prefix: str = '',
                             index: str = '') -> str:
        """
        Generates a graph network based on the given DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the graph information.
        - colored (str): The node to be colored in the graph.
        - index (str, optional): The index of the graph. Defaults to ''.

        Returns:
        - str: The file path of the generated graph image.
        """
        G = Digraph(format='jpeg')
        G.attr(rankdir='LR', size='8,5')
        G.attr('node', shape='circle')
        
        nodelist = []
        for _, row in df.iterrows():
            node1, node2, weight = [str(i) for i in row]
            if float(weight) == 0.0:
                continue
                
            if node1 not in nodelist:
                if node1 == colored:
                    G.node(node1, style='filled', fillcolor='#40e0d0')
                else:
                    G.node(node1)
                nodelist.append(node1)
            
            if node2 not in nodelist:
                if node2 == colored:
                    G.node(node2, style='filled', fillcolor='#40e0d0')
                else:
                    G.node(node2)
                nodelist.append(node2)

            G.edge(node1, node2, label=weight)

        if index != '':
            if int(index) < 10:
                index = f'0{index}'
        fname = f'./markov_chain_{prefix}_{index}_{colored}'
        G.render(fname, view=False)
        save_as = f'{fname}.jpeg'
        return save_as

    def generate_gif(self,
                     jpeg_dir: str = './',
                     file_prefix: str = 'markov_chain',
                     save_as: str = './markov_chain.gif',
                     duration: int = 1000) -> str:
        """
        Generates a GIF from a directory of jpeg images.

        Args:
            jpeg_dir (str): The directory containing the jpeg images. Defaults to './'.
            save_as (str): The path to save the generated GIF. Defaults to './markov_chain.gif'.
            duration (int): The duration of each frame in milliseconds. Defaults to 1000.

        Returns:
            str: The path to the generated GIF.

        Raises:
            FileNotFoundError: If the specified jpeg directory does not exist.

        Notes:
            - Only jpeg images with the '.jpeg' extension (and not ending with '_None.jpeg') will be included in the GIF.
            - Each image will have a step information overlayed on it with the format 
              'STEP: XX', where XX is the number extracted from the image filename.
        """
        images: List[Image.Image] = []
        for file_name in sorted(os.listdir(jpeg_dir)):
            if file_name.startswith(file_prefix):
                if file_name.endswith('.jpeg') and not file_name.endswith('_None.jpeg'):
                    file_path = os.path.join(jpeg_dir, file_name)
                    
                    # add step info
                    img = Image.open(file_path)
                    img_draw = ImageDraw.Draw(img)
                    img_draw.text((10, 10), 'STEP: ' + file_name.split(f'{file_prefix}')[1][:2], fill='black')
                    img.save(file_path)
                    
                    images.append(imageio.imread(file_path))
        imageio.mimsave(save_as, images, 'GIF', duration=duration)
        return save_as

    def clear_all_generated_files(self,
                                  file_prefix: str = 'markov_chain') -> None:
        """
        Clears all generated files in the current directory.
        """
        mydir: str = './'
        filelist: List[str] = [f for f in os.listdir(mydir) if f.startswith(file_prefix)]
        for f in filelist:
            os.remove(os.path.join(mydir, f))

    def travel_simulation(self,
                          init_region: str,
                          dest_region: str,
                          plot_mode: str='base') -> List[str]:
        """
        Simulates a travel itinerary based on a given DataFrame of regions and their probabilities of transition.

        Parameters:
            plot_mode (str, optional): The plot mode. Defaults to 'base'.
                                       Options: ['base', 'blocked']
            init_region (str): The initial region to start the travel simulation from.
            dest_region (str): The destination region to reach in the travel simulation.

        Returns:
            List[str]: A list of regions representing the simulated travel itinerary.
        """
        if plot_mode == 'base':
            selected_df = self.base_df
        elif plot_mode == 'blocked':
            selected_df = self.blocked_df
            
        self.travel_simulated = []

        region = init_region
        self.travel_simulated.append(region)

        while region != dest_region:
            region = np.random.choice(selected_df.iloc[selected_df.index.get_loc(region)].index,
                                    p=selected_df.iloc[selected_df.index.get_loc(region)])
            self.travel_simulated.append(region)
        return self.travel_simulated


    def matrix_power(self,
                     matrix: np.ndarray,
                     power: int) -> np.ndarray:
        """
        Calculate the power of a matrix.

        Args:
            matrix (np.ndarray): The matrix to be raised to a power.
            power (int): The power to raise the matrix to.

        Returns:
            np.ndarray: The resulting matrix after raising it to the specified power.
        """
        if power == 0:
            return np.identity(len(matrix))
        elif power == 1:
            return matrix
        else:
            return np.dot(matrix, self.matrix_power(matrix, power-1))

    def prob_ending_region_after_n_step(self, matrix_df: pd.DataFrame,
                                        init_region: str,
                                        final_region: str,
                                        n_step: int) -> float:
        """
        Calculates the probability of ending in a specific region after taking a certain number of steps in a matrix.

        Args:
            matrix_df (pd.DataFrame): The input matrix containing transition probabilities between regions.
            init_region (str): The initial region.
            final_region (str): The final region.
            n_step (int): The number of steps to take.

        Returns:
            float: The probability of ending in the final region after n steps.
        """
        region_list = list(matrix_df.columns)
        initial_dist = np.asarray([i==init_region for i in region_list])
        df_trip_2 = self.matrix_power(matrix_df.to_numpy(),n_step)
        res = np.dot(initial_dist,df_trip_2)
        final_region_idx = region_list.index(final_region)
        prob_ending_in_final_region_after_n_step = res[final_region_idx]
        return prob_ending_in_final_region_after_n_step 
    
    def plot_base(self):
        """
        Generate the base plot for visualization.

        Clears all previously generated files.

        Parameters:
            None

        Returns:
            The generated base plot image.
        """
        # clear previous  all generated files
        self.clear_all_generated_files(file_prefix='markov_chain_base')

        # declare matrix
        self.base_df = self.baseline_data(self.region_list,
                                          self.remove_self_flight)
        self.prep_base_df = self.preprocess_data(self.base_df)

        print('BASELINE MATRIX ')
        print(self.base_df)
        
        # render graph network
        self.base_img = self.create_graph_network(self.prep_base_df,
                                                  colored=None,
                                                  prefix='base')
        return self.base_img

    def plot_blocked_node(self,
                          node_1: str,
                          node_2: str):
        """
        Plot the blocked node between two given nodes.

        Parameters:
            node_1 (str): The first node.
            node_2 (str): The second node.

        Returns:
            blocked_img: The graph network image after blocking the nodes.
        """
        # block nodes
        self.blocked_df = self.block_nodes(self.base_df, 
                                            node_1, 
                                            node_2, 
                                            self.remove_self_flight)
        print(f'\n BLOCKED MATRIX BETWEEN {node_1} AND {node_2}')
        print(self.blocked_df)

        # preprocess data
        self.prep_blocked_df = self.preprocess_data(self.blocked_df)

        # render graph network
        self.blocked_img = self.create_graph_network(self.prep_blocked_df, 
                                                     colored=None)
        return self.blocked_img
    
    def plot_travel_simulation(self,
                               plot_mode: str = 'base'):
        """
        Plot the travel simulation.

        Parameters:
            plot_mode (str, optional): The plot mode. Defaults to 'base'.
                                       Options: ['base', 'blocked']

        Returns:
            str: The path to the generated travel image.
        """
        # clear previous  all generated files
        self.clear_all_generated_files(file_prefix='markov_chain_blocked')

        # render graph network
        colored_list = self.travel_simulated
        if plot_mode == 'base':
            selected_df = self.base_df
        elif plot_mode == 'blocked':
            selected_df = self.blocked_df
            
        prep_selected_df = self.preprocess_data(selected_df)
        
        for i in range(len(colored_list)):
            self.create_graph_network(prep_selected_df, colored_list[i], plot_mode, index=str(i))
        
        # generate gif
        travel_img = self.generate_gif(file_prefix=f'markov_chain_{plot_mode}_', 
                                       save_as= './markov_chain_blocked_travel_sim.gif')
        return travel_img