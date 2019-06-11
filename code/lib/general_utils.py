'''
    Defines general utility functions
'''
from models.architectures import ALLOWABLE_TYPES as ALLOWABLE_MODEL_TYPES

import os 


#############################
#  Dynamically set variables
#############################
class RuntimeDeterminedEnviromentVars( object ):
    '''
        Example use:
        inputs = { 'num_samples_epoch': 100 }
        cfg = { 'batch_size': 5, 'epoch_steps': [ '<LOAD_DYNAMIC>', 'steps_per_epoch' ] }

        for key, value in cfg.items():
            if isinstance( value, list ) and len( value ) == 2 and value[0] == 'LOAD_DYNAMIC':
                RuntimeDeterminedEnviromentVars.register( cfg, key, value[1] )

        RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        RuntimeDeterminedEnviromentVars.populate_registered_variables()
        print( cfg )  # epoch_steps = 20
    '''
    registered_variables = []
    is_loaded = False
    # These are initialized in load_dynamic_variables
    steps_per_epoch = ''  # An int that condains the number of steps the network will take per epoch
    
    @classmethod
    def load_dynamic_variables( cls, inputs, cfg ):
        '''
            Args:
                inputs: a dict from train.py
                cfg: a dict from a config.py
        '''    
        cls.steps_per_epoch = inputs[ 'num_samples_epoch' ] // cfg[ 'batch_size' ]
        cls.is_loaded = True
    
    @classmethod
    def register( cls, dict_containing_field_to_populate, field_name, attr_name ):
        cls.registered_variables.append( [dict_containing_field_to_populate, field_name, attr_name] )
    
    @classmethod
    def register_dict( cls, dict_to_register ):
        '''
            Registers any fields in the dict that should be dynamically loaded.
            Such fields should have value: [ '<LOAD_DYNAMIC>', attr_name ]
        '''
        for key, value in dict_to_register.items():
            if isinstance( value, list ) and len( value ) == 2 and value[0] == '<LOAD_DYNAMIC>':
                cls.register( dict_to_register, key, value[1] )
            elif isinstance( value, dict ):
                cls.register_dict( value )

    @classmethod
    def populate_registered_variables( cls ):
        print( "dynamically populating variables:" )
        for dict_containing_field_to_populate, field_name, attr_name in cls.registered_variables:
            dict_containing_field_to_populate[field_name] = getattr( cls, attr_name )
            print( "\t{0}={1}".format( field_name, getattr( cls, attr_name ) ) )


###########################
#  Utility functions
###########################
def validate_config( cfg ):
    '''
        Catches many general cfg errors. 
    '''
    if  cfg[ 'model_type' ] not in ALLOWABLE_MODEL_TYPES:
        raise ValueError( "'model_type' in config.py must be one of {0}".format( ALLOWABLE_MODEL_TYPES ))
    if cfg[ 'model_type' ] is not 'empty' and 'optimizer' not in cfg:
        raise ValueError( "an 'optimizer' must be specified".format( ALLOWABLE_MODEL_TYPES ))
    if 'optimizer' in cfg and 'optimizer_kwargs' not in cfg:
        raise ValueError( "The arguments for the optimizer {0} must be given, named, in 'optimizer_kwargs'".format( cfg[ 'optimizer' ] ))


def load_config( cfg_dir, nopause=False ):
    ''' 
        Raises: 
            FileNotFoundError if 'config.py' doesn't exist in cfg_dir
    '''
    if not os.path.isfile( os.path.join( cfg_dir, 'config.py' ) ):
        raise ImportError( 'config.py not found in {0}'.format( cfg_dir ) )
    import sys
    try:
        del sys.modules[ 'config' ]
    except:
        pass

    sys.path.insert( 0, cfg_dir )
    import config as loading_config
    # cleanup
    # print([ v for v in sys.modules if "config" in v])
    # return
    cfg = loading_config.get_cfg( nopause )

    try:
        del sys.modules[ 'config' ]
    except:
        pass
    sys.path.remove(cfg_dir)

    return cfg

def update_keys(old_dict, key_starts_with, new_dict):
    for k, v in new_dict.items():
        if k.startswith(key_starts_with):
            old_dict[k] = v
    return old_dict