#include "DataHandler.h"
#include "ArgumentReader.h"


std::default_random_engine g_rand;
size_t g_max_threads;

int main( int argc, char** argv )
{
  uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
  g_rand = std::default_random_engine( seed );

  try {
    InputArgs inps( argc, argv );
    //std::cout << inps.toString() << std::endl;

    g_max_threads = inps.getArg<int>( "-t" );

    std::cout << "SAVING TO: " << inps.getArg<std::string>( "-p" ) << std::endl;
    npy::DataHandler<float> dth( inps.getArg<std::string>( "-p" ), inps.getArg<npy::Coordinate>( "-s" ) );
    std::cout << "Created Volume" << std::endl;

    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
    int freq = inps.getArg<double>( "-f" ); double range = inps.getArg<double>( "-r" );
    int g_perlin_iter = inps.getArg<int>( "-i" ); double divs = inps.getArg<double>( "-d" );
    std::cout << "Starting Perlin Generation" << std::endl;
    for( size_t it=0; it < g_perlin_iter; ++it )
    {
        dth.addPerlinNoise( npy::Coordinate( freq,0,0 ), npy::Range<double>( -range, range ) );
        freq *= divs;
        range /= divs;
    }
    std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> r_time_perlin = std::chrono::duration_cast<std::chrono::duration<double>>( t_end -t_start );

    return 0;
  }
  catch( std::invalid_argument e ) {
    std::cout << "Invalid argument exception thrown: " << e.what() << std::endl;
    return -1;
  }

}
