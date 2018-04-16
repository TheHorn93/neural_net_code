#include "DataHandler.h"
#include <windows.h>


std::string g_output_path;
npy::Range<size_t> g_num_gauss;
npy::Range<double> g_scale, g_std_div, g_perlin_range;
double g_perlin_freq;
size_t g_perlin_iter;
npy::Range<double> g_ppv_noise;
npy::Coordinate g_data_size;
std::default_random_engine g_rand;

bool parseCmdArgs( int _argc, char** _argv )
{
    if( _argc != 10 )
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
        else if ( arg_name == "perlin_freq" )
        {
            g_perlin_freq = std::stof( arg_val );
            std::cout << "Set \"" << arg_name << "\" to \"" << g_perlin_freq << "\"" << std::endl;
        }
        else if ( arg_name == "perlin_range" )
        {
            g_perlin_range = arg_val;
            std::cout << "Set \"" << arg_name << "\" to \"" << g_perlin_range << "\"" << std::endl;
        }
        else if ( arg_name == "perlin_iter" )
        {
            g_perlin_iter = std::stoi( arg_val );
            std::cout << "Set \"" << arg_name << "\" to \"" << g_perlin_iter << "\"" << std::endl;
        }
        else if ( arg_name == "ppv_noise" )
        {
            g_ppv_noise = arg_val;
            std::cout << "Set \"" << arg_name << "\" to \"" << g_ppv_noise << "\"" << std::endl;
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

    /*npy::Volume<float> vol( npy::Coordinate( 3,6,4 ) ), vol_pad;
    size_t it = 0;
    for( size_t x=0; x < 3; ++x )
    {
        for( size_t y=0; y < 6; ++y )
        {
            for( size_t z=0; z < 4; ++z )
            {
                vol( npy::Coordinate( x,y,z ) ) = it;
                ++it;
            }
        }
    }
    vol_pad.padFromVolume( vol, 2 );
    vol.addWithOffset( vol_pad, npy::Coordinate( 2,2,2 ) );
    std::cout << vol << " with " << vol.shape() << std::endl;
    std::cout << vol_pad << std::endl;*/

    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    g_rand = std::default_random_engine( seed );
    srand( time( NULL ) );

    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
    dth.addSmallGaussians( g_num_gauss, g_scale, g_std_div );
    std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> r_time_gauss = std::chrono::duration_cast<std::chrono::duration<double>>( t_end -t_start );

    t_start = std::chrono::high_resolution_clock::now();
    int freq = g_perlin_freq; double range = g_perlin_range.getRandDouble();
    std::cout << range << std::endl;
    for( size_t it=0; it < g_perlin_iter; ++it )
    {
        dth.addPerlinNoise( npy::Coordinate( freq,0,0 ), npy::Range<double>( -range, range ) );
        freq *= 2;
        range /= 2;
    }
    t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> r_time_perlin = std::chrono::duration_cast<std::chrono::duration<double>>( t_end -t_start );

    t_start = std::chrono::high_resolution_clock::now();
    dth.addGuidedPerVoxelUniformNoise( g_ppv_noise );
    t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> r_time_ppv = std::chrono::duration_cast<std::chrono::duration<double>>( t_end -t_start );

    std::cout << "Gaussian: " << r_time_gauss.count() << ", iied Noise: " << r_time_ppv.count() << ", Perlin: " << r_time_perlin.count() << std::endl;
    return 0;
}
