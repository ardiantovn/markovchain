import streamlit as st

from markovchain import MarkovChain

class StreamlitMC:
    def add_region_list(self):
        
        instr = 'Enter  regions (comma separated)'

        with st.form('chat_input_form'):
            # Create two columns; adjust the ratio to your liking
            col1, col2 = st.columns([3,1]) 

            # Use the first column for text input
            with col1:
                prompt = st.text_input(
                    instr,
                    placeholder=instr,
                    label_visibility='collapsed'
                )
            # Use the second column for the submit button
            with col2:
                submitted = st.form_submit_button('ENTER')
            
            if prompt and submitted:
                # Do something with the inputted text here
                st.write(f"Region List: {prompt}")
                
        self.region_list = prompt.split(',')


    def button_remove_self_flight(self):
        # Initialize the "Remove Self Flight" option
        self.remove_self_flight = False

        # Display a power button to toggle the "Remove Self Flight" option
        self.remove_self_flight = st.sidebar.toggle("Remove Self Flight[Optional]", self.remove_self_flight)

if __name__ == "__main__":
    '''
    # UAV Markov Chain Simulator
    A UAV has been programmed to move between several region. 
    This simulator will generate possible routes for the UAV based on the inputted region list.
    The UAV movement is determined by the Markov Chain using random transition matrix.
    
    ## Add Region List
    Example: Gunja, Wangsimni, Dapsimni
    '''
    
    obj = StreamlitMC()
    obj.add_region_list()
    
    if len(obj.region_list)>1:
        st.sidebar.title("Plot Basic Transision Matrix")
        plot_base = False
        plot_base = st.sidebar.toggle("Plot Basic", plot_base)
        
        obj.button_remove_self_flight()
        
        remove_updated = False
        if 'remove_self_flight' not in st.session_state:
            st.session_state.remove_self_flight = obj.remove_self_flight
        elif obj.remove_self_flight != st.session_state.remove_self_flight:
            st.session_state.remove_self_flight = obj.remove_self_flight
            remove_updated = True
            
        if plot_base:
            mc = MarkovChain(obj.region_list,
                             st.session_state.remove_self_flight)
            
            if 'mc' not in st.session_state or remove_updated==True:
                st.session_state.mc = mc
                st.session_state.mc.plot_base()
                
                st.session_state.base_img = st.session_state.mc.base_img
                st.session_state.base_df = st.session_state.mc.base_df
            
            # base_img = mc.plot_base()
            
            '''
            ## Basic Transision Matrix and its Graph Network
            '''
            base1, base2 = st.columns(2)
            with base1:
                st.dataframe(st.session_state.mc.base_df)
            with base2:
                st.image(st.session_state.mc.base_img)

            st.sidebar.title('UAV Flight Simulation')
            st.sidebar.text('Choose Departure and Arrival Region.')
            
            start = st.sidebar.selectbox('Departure Region', obj.region_list)
            end = st.sidebar.selectbox('Arrival Region', obj.region_list)
            
            plot_blocked = False
            plot_blocked = st.sidebar.toggle("Plot Route", plot_blocked)
            
            if plot_blocked:
                '''
                ## UAV Flight Transision Matrix and its Graph Network
                '''
                if 'blocked' not in st.session_state:
                    st.session_state.blocked = True
                    st.session_state.mc.plot_blocked_node(node_1=start,
                                                                    node_2=end)
                    
                    st.session_state.blocked_img = st.session_state.mc.blocked_img
                    st.session_state.blocked_df = st.session_state.mc.blocked_df
                
                block1, block2 = st.columns(2)
                with block1:
                    st.dataframe(st.session_state.mc.blocked_df)
                with block2:
                    st.image(st.session_state.mc.blocked_img)
                
                simulate = st.button('SIMULATE')
                
                if simulate:
                    plot_mode='blocked'
                    travel_simulated = st.session_state.mc.travel_simulation(start,
                                                                            end,
                                                                            plot_mode
                                                                            )
                    f'''
                    ## SIMULATION RESULTS
                    - N-STEP : {len(travel_simulated)-1}
                    - POSSIBLE ROUTE: {travel_simulated}
                    '''
                    
                    travel_img = st.session_state.mc.plot_travel_simulation(plot_mode)
                    st.image(travel_img)
                    
                
                
                    