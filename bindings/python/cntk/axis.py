<class 'cntk.cntk_py.Axis'>
all_static_axes
default_batch_axis
default_dynamic_axis
    @typemap
    def is_ordered(self):
        '''
        is_ordered
        

        Returns:
            `None`: text
        '''
        return super(Axis, self).is_ordered()

    @typemap
    def is_static_axis(self):
        '''
        is_static_axis
        

        Returns:
            `None`: text
        '''
        return super(Axis, self).is_static_axis()

    @typemap
    def name(self):
        '''
        name
        

        Returns:
            `None`: text
        '''
        return super(Axis, self).name()

new_unique_dynamic_axis
    @typemap
    def static_axis_index(self, checked=True):
        '''
        static_axis_index
        

        Args:
            d (`checke`): text
        

        Returns:
            `None`: text
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(Axis, self).static_axis_index(**kwargs)

