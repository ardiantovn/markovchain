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
    Example: Ttukseom, Gunja, Wangsimni, Dapsimni
    '''
    
    obj = StreamlitMC()
    obj.add_region_list()
    region_updated = False
    if 'region_list' not in st.session_state:
        st.session_state.region_list = obj.region_list
    elif obj.region_list != st.session_state.region_list:
        st.session_state.region_list = obj.region_list
        region_updated = True
    
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
            mc = MarkovChain(st.session_state.region_list,
                             st.session_state.remove_self_flight)
            
            if 'mc' not in st.session_state\
                or remove_updated\
                or region_updated:
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
            
            node_1_updated = False
            if 'start' not in st.session_state:
                st.session_state.start = start
            elif start != st.session_state.start:
                st.session_state.start = start
                node_1_updated = True
                
            node_2_updated = False
            if 'end' not in st.session_state:
                st.session_state.end = end
            elif end != st.session_state.end:
                st.session_state.end = end
                node_2_updated = True
            
            plot_blocked = False
            plot_blocked = st.sidebar.toggle("Plot Route", plot_blocked)
            
            if plot_blocked:
                '''
                ## UAV Flight Transision Matrix and its Graph Network
                '''
                if 'blocked' not in st.session_state \
                    or remove_updated \
                    or region_updated \
                    or node_1_updated \
                    or node_2_updated:
                    st.session_state.blocked = True
                    st.session_state.mc.plot_blocked_node(node_1=st.session_state.start,
                                                          node_2=st.session_state.end)
                    
                    st.session_state.blocked_img = st.session_state.mc.blocked_img
                    st.session_state.blocked_df = st.session_state.mc.blocked_df
                
                block1, block2 = st.columns(2)
                with block1:
                    st.dataframe(st.session_state.mc.blocked_df)
                with block2:
                    st.image(st.session_state.mc.blocked_img)
                
                f'''
                ## SIMULATION
                
                This section will simulate the possible routes for the UAV starting
                from {start.upper()} and ending in {end.upper()}. The first result will
                generate the route randomly based on the transition matrix.
                
                In the second result, it will show the probability of ending in {end.upper()}
                after N-step. You can change the N value using the slider below.
                '''
                
                n_step = st.slider('### How many step?', 0, 50, 20)
                
                simulate = st.button("SIMULATE")
                
                if simulate:
                    plot_mode='blocked'
                
                    travel_simulated = st.session_state.mc.travel_simulation(start,
                                                                            end,
                                                                            plot_mode
                                                                            )
                    
                    f'''
                    ## SIMULATION RESULTS
                    
                    The UAV finished the trip after **{len(travel_simulated)-1} STEP** through
                    this **ROUTE: {travel_simulated}**.
                    '''
                    travel_img = st.session_state.mc.plot_travel_simulation(plot_mode)
                    
                    st.image(travel_img)
                    
                    f'''
                    ## Probability Ending in {end.upper()} After N-STEP
                    
                    This plot below shows the probability of ending in {end.upper()}
                    {n_step}-step. 
                    '''
                    
                    plot_prob = st.session_state.mc.plot_prob_ending(init_region=start,
                                                                    final_region=end,
                                                                    matrix_df=st.session_state.mc.blocked_df,
                                                                    n_step=n_step)
                    
                    st.image(plot_prob)
                    
                
                
                    