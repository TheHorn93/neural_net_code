#include "DataHandler.h"
#include <windows.h>

std::string g_output_path;
npy::Range<size_t> g_num_gauss;
npy::Range<double> g_scale, g_std_div;
npy::Coordinate g_data_size;

bool parseCmdArgs( int _argc, char** _argv )
{
    if( _argc != 6 )
    {
        std::cerr << "Wrong number of arguments" << std::endl;
        return false;
    }

    for( int it=1; it < _argc; ++it )
    {
        std::string arg( _argv[it] );
        std::string arg_name = arg.substr( 0, arg.find( '=' ) );
        std::string arg_val = arg.substr( arg.find( '=' ) +1 );

        if( arg_name == "output_path" )
        {
            g_output_path = arg_val;
            std::cout << "Set \"" << arg_name << "\" to \"" << g_output_path << "\"" << std::endl;
        }
        else if ( arg_name == "data_size" )
        {
            g_data_size = arg_val;
            std::cout << "Set \"" << arg_name << "\" to \"" << g_data_size << "\"" << std::endl;
        }
        else if ( arg_name == "num_gauss" )
        {
            g_num_gauss = arg_val;
            std::cout << "Set \"" << arg_name << "\" to \"" << g_num_gauss << "\"" << std::endl;
        }
        else if ( arg_name == "scale" )
        {
            g_scale = arg_val;
            std::cout << "Set \"" << arg_name << "\" to \"" << g_scale << "\"" << std::endl;
        }
        else if ( arg_name == "std_div" )
        {
            g_std_div = arg_val;
            std::cout << "Set \"" << arg_name << "\" to \"" << g_std_div << "\"" << std::endl;
        }
        else
        {
            std::cerr << "Wrong Argument Name" << std::endl;
            return false;
        }
    }
    return true;
}

int main( int argc, char** argv )
{
    if( !parseCmdArgs( argc, argv ) )
        return -1;

    npy::DataHandler<float> dth( g_output_path, g_data_size );

    dth.addMultipleGaussians( g_num_gauss, g_scale, g_std_div );

    return 0;
}
