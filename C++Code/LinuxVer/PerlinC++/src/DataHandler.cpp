#include "DataHandler.h"

extern size_t g_max_threads;
namespace npy
{
#ifndef MAX_THREADS__
#define MAX_THREADS__ 4
#endif // MAX_THREADS__

#define FWHM 5

template <typename _Tp>
DataHandler<_Tp>::DataHandler( const std::string& _i_file_path, const std::string& _o_file_path, npy::Coordinate _shape  )
:
    m_shape( _shape ),
    m_size( 1 )
{
    for( size_t it=0; it < 3; ++it )
        m_size *= m_shape[it];
    m_data = Volume<_Tp>( m_shape );
    m_f_handler.reset( new FileHandler<_Tp>( _i_file_path, _o_file_path ) );
    m_f_handler->loadFile( m_data.data().data() );
}



template <typename _Tp>
DataHandler<_Tp>::DataHandler( const std::string& _o_file_path, npy::Coordinate _shape  )
:
    m_shape( _shape ),
    m_size( 1 )
{
    for( size_t it=0; it < 3; ++it )
        m_size *= m_shape[it];
    m_data = Volume<_Tp>( m_shape );
    m_f_handler.reset( new FileHandler<_Tp>( _o_file_path ) );
    std::cout << "Allocated " << m_size << " elements" << std::endl;
}



template <typename _Tp>
DataHandler<_Tp>::~DataHandler()
{
    std::cout << "Saving File to: " << m_f_handler->getOutPath() << std::endl;
    m_f_handler->saveFile( m_data.data().data(), m_size );
    std::cout << "Freeing Buffer" << std::endl;
}


template <typename _Tp>
void DataHandler<_Tp>::evalGauss( const size_t _max_div, const double _iter )
{
    size_t max_iter = _max_div * FWHM / _iter;
    std::vector<double> output;
    output.reserve( max_iter );

    //float scale_iter = 0.0;
    float variance = std::pow( _max_div, 2 );
    for( size_t it=0; it < max_iter; ++it )
    {
        double expo = - std::pow( it *_iter, 2 ) / ( 2 * variance );
        output.push_back( exp( expo ) );
    }
    m_gauss_vals = output;
}



template <typename _Tp>
void DataHandler<_Tp>::addGaussianThread( const size_t _max_range, const double _g_idx )
{
    npy::Coordinate voxel( 0, 0, 0 );
    while( true )
    {
        voxel[1] = 0;
        voxel[2] = 0;
        voxel[0] = getNextIter();
        if( voxel[0] >= m_shape[0] )
            break;

        size_t mem_it = voxel[0] *m_thread_step;
        do
        {
            //m_data[mem_it] = 0;
            for( size_t g_it=0; g_it < m_gaussians.size(); ++g_it )
            {
                size_t sq_diff = voxel.getSqrdDistance( ( m_gaussians[g_it] ).center );
                if ( sq_diff <= m_gaussians[g_it].sqrt_rad )
                {
                    double proto_it = ( std::sqrt( sq_diff ) / m_gaussians[g_it].rad ) * _max_range / _g_idx;
                    double weight = std::ceil( proto_it ) - proto_it;
                    m_data[mem_it] += m_gaussians[g_it].scale *( weight * m_gauss_vals[std::floor( proto_it )] + (1-weight) * m_gauss_vals[std::ceil( proto_it )] );
                }
            }
            ++mem_it;
        } while( iterateCoordinate( voxel ) );
    }

    std::cout << std::this_thread::get_id() << " finished" << std::endl;
    return;
}



template <typename _Tp>
void DataHandler<_Tp>::createStructureGaussians( const Range<size_t>& _num_gauss,
                                                 const Range<double>& _std_div_r,
                                                 std::vector<Gaussian>& _output )
{
    size_t max_range = std::ceil( _std_div_r[1]*FWHM );
    double max_rad = _std_div_r[1] * FWHM;
    double g_idx = 0.0001;
    evalGauss( _std_div_r[1], g_idx );
    m_iter = -1;
    m_thread_step = m_shape[1] *m_shape[2];

    std::vector<Gaussian> str_gauss;
    size_t num_gauss = _num_gauss.getRandInt();
    for( size_t it=0; it < num_gauss; ++it )
    {
        Coordinate center;
        do
        {
            std::vector<int> coord;
            for( int it=0; it < 3; ++it )
                coord.push_back( rand()%m_shape[it] );
            center = Coordinate( coord );
        }
        while( CoordInList( center ) );
        str_gauss.push_back( Gaussian( center, 1.0, _std_div_r.getRandDouble() * FWHM, max_rad ) );
        m_coords.push_back( center );
    }
    m_coords.clear();

    _output = str_gauss;
}



template <typename _Tp>
void DataHandler<_Tp>::addMultipleGaussians( const Range<size_t>& _num_gauss,
                                             const Range<double>& _scale_r,
                                             const Range<double>& _std_div_r )
{
    size_t max_range = std::ceil( _std_div_r[1]*FWHM );
    double max_rad = _std_div_r[1] * FWHM;
    double g_idx = 0.0001;
    if( max_range < 100 )
        g_idx = 0.000001;
    evalGauss( _std_div_r[1], g_idx );
    m_iter = -1;
    m_thread_step = m_shape[1] *m_shape[2];

    size_t num_gauss = _num_gauss.getRandInt();
    for( size_t it=0; it < num_gauss; ++it )
    {
        Coordinate center;
        do
        {
            std::vector<int> coord;
            for( int it=0; it < 3; ++it )
                coord.push_back( rand()%m_shape[it] );
            center = Coordinate( coord );
        }
        while( CoordInList( center ) );
        m_gaussians.push_back( Gaussian( center, _scale_r.getRandDouble(), _std_div_r.getRandDoubleNonLinear() * FWHM, max_rad ) );
        m_coords.push_back( center );
    }

    for( size_t it=0; it < MAX_THREADS__; ++it )
        m_threads.push_back( std::thread( &DataHandler::addGaussianThread, this, max_range, g_idx ) );

    for( size_t it=0; it < MAX_THREADS__; ++it )
        m_threads[it].join();

    m_threads.clear();
    m_gaussians.clear();
    m_coords.clear();

    std::cout << "Threads joined" << std::endl;
}



template <typename _Tp>
void DataHandler<_Tp>::addStructuredGaussians( const Range<size_t>& _num_structures,
                                               const Range<double>& _str_std_div,
                                               const Range<size_t>& _num_gauss,
                                               const Range<double>& _scale_r,
                                               const Range<double>& _std_div_r )
{
    std::vector<Gaussian> str_gaussians;
    createStructureGaussians( _num_structures, _str_std_div, str_gaussians );

    size_t max_range = std::ceil( _std_div_r[1]*FWHM );
    double max_rad = _std_div_r[1] * FWHM;
    double g_idx = 0.0001;
    evalGauss( _std_div_r[1], g_idx );
    m_iter = -1;
    m_thread_step = m_shape[1] *m_shape[2];

    size_t num_gauss = _num_gauss.getRandInt();
    for( size_t it=0; it < num_gauss; ++it )
    {
        Coordinate center;
        do
        {
            std::vector<int> coord;
            for( int it=0; it < 3; ++it )
                coord.push_back( rand()%m_shape[it] );
            center = Coordinate( coord );
        }
        while( CoordInList( center ) );
        m_gaussians.push_back( Gaussian( center, _scale_r.getRandDouble(), getStrStdDiv( center, _std_div_r, str_gaussians ) * FWHM, max_rad ) );
        m_coords.push_back( center );
        if( it %100 == 0 )
            std::cout << "Created Gaussian " << it << std::endl;
    }

    for( size_t it=0; it < MAX_THREADS__; ++it )
        m_threads.push_back( std::thread( &DataHandler::addGaussianThread, this, max_range, g_idx ) );

    for( size_t it=0; it < MAX_THREADS__; ++it )
        m_threads[it].join();

    m_threads.clear();
    m_gaussians.clear();
    m_coords.clear();

    std::cout << "Threads joined" << std::endl;
}



template <typename _Tp>
void DataHandler<_Tp>::addSmallGaussianThread( const size_t _max_range,
                                               const size_t _it_sta,
                                               const size_t _it_end )
{
    Volume<_Tp> mask( m_shape );
    //mask.padFromVolume( m_data, _max_range );

    size_t it_end = std::min( m_gaussians.size(), _it_end );
    {
        std::unique_lock<std::mutex> iter_lock( m_iter_mutex );
        std::cout << it_end << " of " << m_gaussians.size() << std::endl;
    }
    size_t x_step = m_shape[1] *m_shape[2]; size_t y_step = m_shape[2];

    for( size_t it=_it_sta; it < it_end; ++it )
    {
        int proto_max_it = std::floor( m_gaussians[it].rad ) *FWHM;
        //m_gaussians[it].center = m_gaussians[it].center +_max_range;
        npy::Coordinate mem_start( std::max( 0, m_gaussians[it].center[0] -proto_max_it ),
                                   std::max( 0, m_gaussians[it].center[1] -proto_max_it ),
                                   std::max( 0, m_gaussians[it].center[2] -proto_max_it ) );
        npy::Coordinate mem_end( std::min( m_shape[0], m_gaussians[it].center[0] +proto_max_it ),
                                 std::min( m_shape[1], m_gaussians[it].center[1] +proto_max_it ),
                                 std::min( m_shape[2], m_gaussians[it].center[2] +proto_max_it ) );

        int max_it = proto_max_it +1;
        std::vector<_Tp> mask_1d;
        mask_1d.resize( max_it *2 -1 );
        int mean = max_it -1;
        for( size_t jt=0; jt < max_it; ++jt )
        {
            double expo = -std::pow( jt, 2 ) / ( 2 *m_gaussians[it].sqrt_rad );
            double val = exp( expo );
            mask_1d[mean -jt] = val;
            mask_1d[mean +jt] = val;
            //std::cout << val << " from " << m_gaussians[it].sqrt_rad << " -> " << expo <<  std::endl;
        }
        npy::Coordinate gauss_offset( -m_gaussians[it].center[0] +proto_max_it +mem_start[0],
                                      -m_gaussians[it].center[1] +proto_max_it +mem_start[1],
                                      -m_gaussians[it].center[2] +proto_max_it +mem_start[2] );
        npy::Coordinate gauss_iter = gauss_offset;
        if( it %25000 == 0 )
        {
            std::unique_lock<std::mutex> iter_lock( m_iter_mutex );
            std::cout << gauss_iter << " = " << m_gaussians[it].center << " + " << mem_start << std::endl;
        }
        for( size_t x_it=mem_start[0]; x_it < mem_end[0]; ++x_it )
        {
            size_t mem_iter_x = x_step *x_it;
            for( size_t y_it=mem_start[1]; y_it < mem_end[1]; ++y_it )
            {
                size_t mem_iter_y = y_step *y_it;
                for( size_t z_it=mem_start[2]; z_it < mem_end[2]; ++z_it )
                {
                    size_t mem_iter = mem_iter_x + mem_iter_y + z_it;
                    double val = mask_1d[gauss_iter[0]] *mask_1d[gauss_iter[1]] *mask_1d[gauss_iter[2]] *m_gaussians[it].scale;
                    mask[mem_iter] += val;
                    ++gauss_iter[2];
                    //{std::unique_lock<std::mutex> iter_lock( m_iter_mutex );
                    //std::cout << std::this_thread::get_id() << " Mem:" << mem_iter << " from " << x_it << ", " << y_it << ", "<< z_it << std::endl;}
                }
                gauss_iter[2] = gauss_offset[2];
                ++gauss_iter[1];
            }
            gauss_iter[1] = gauss_offset[1];
            ++gauss_iter[0];
        }
    }
    std::unique_lock<std::mutex> iter_lock( m_iter_mutex );
    m_data.addWithOffset( mask, Coordinate( 0,0,0 ) );
}



/*template <typename _Tp>
void DataHandler<_Tp>::addSmallGaussianThread( const size_t _max_range,
                                               const size_t _it_sta,
                                               const size_t _it_end )
{
    Volume<_Tp> mask;
    mask.padFromVolume( m_data, _max_range );

    size_t it_end = std::min( m_gaussians.size(), _it_end );
    {
        std::unique_lock<std::mutex> iter_lock( m_iter_mutex );
        std::cout << it_end << " of " << m_gaussians.size() << std::endl;
    }
    size_t x_step = mask.getXStep(); size_t y_step = mask.shape()[2];

    for( size_t it=_it_sta; it < it_end; ++it )
    {
        int proto_max_it = std::floor( m_gaussians[it].rad ) *FWHM;
        m_gaussians[it].center = m_gaussians[it].center +_max_range;
        npy::Coordinate mem_start( m_gaussians[it].center[0] -proto_max_it,
                                   m_gaussians[it].center[1] -proto_max_it,
                                   m_gaussians[it].center[2] -proto_max_it );
        npy::Coordinate mem_end( m_gaussians[it].center[0] +proto_max_it,
                                 m_gaussians[it].center[1] +proto_max_it,
                                 m_gaussians[it].center[2] +proto_max_it );

        int max_it = proto_max_it +1;
        std::vector<_Tp> mask_1d;
        mask_1d.resize( max_it *2 -1 );
        int mean = max_it -1;
        for( size_t jt=0; jt < max_it; ++jt )
        {
            double expo = -std::pow( jt, 2 ) / ( 2 *m_gaussians[it].sqrt_rad );
            double val = exp( expo );
            mask_1d[mean -jt] = val;
            mask_1d[mean +jt] = val;
        }

        npy::Coordinate gauss_iter( 0,0,0 );
        if( it %25000 == 0 )
        {
            std::unique_lock<std::mutex> iter_lock( m_iter_mutex );
            std::cout << gauss_iter << " = " << m_gaussians[it].center << " + " << mem_start << std::endl;
        }
        for( size_t x_it=mem_start[0]; x_it < mem_end[0]; ++x_it )
        {
            size_t mem_iter_x = x_step *x_it;
            for( size_t y_it=mem_start[1]; y_it < mem_end[1]; ++y_it )
            {
                size_t mem_iter_y = y_step *y_it;
                for( size_t z_it=mem_start[2]; z_it < mem_end[2]; ++z_it )
                {
                    size_t mem_iter = mem_iter_x + mem_iter_y + z_it;
                    double val = mask_1d[gauss_iter[0]] *mask_1d[gauss_iter[1]] *mask_1d[gauss_iter[2]];
                    mask[mem_iter] += val;
                    ++gauss_iter[2];
                }
                gauss_iter[2] = 0;
                ++gauss_iter[1];
            }
            gauss_iter[1] = 0;
            ++gauss_iter[0];
        }
    }
    std::unique_lock<std::mutex> iter_lock( m_iter_mutex );
    m_data.addWithOffset( mask, Coordinate( _max_range, _max_range, _max_range ) );
}*/



template <typename _Tp>
void DataHandler<_Tp>::addSmallGaussians( const Range<size_t>& _num_gauss,
                                          const Range<double>& _scale_r,
                                          const Range<double>& _std_div_r )
{
    size_t max_range = std::ceil( _std_div_r[1]*FWHM );
    double max_rad = _std_div_r[1] * FWHM;

    size_t num_gauss = _num_gauss.getRandInt();
    std::cout << "Adding " << num_gauss << " gaussians" << std::endl;
    for( size_t it=0; it < num_gauss; ++it )
    {
        Coordinate center;
        std::vector<int> coord;
        for( int it=0; it < 3; ++it )
            coord.push_back( rand()%m_shape[it] );
        center = Coordinate( coord );

        m_gaussians.push_back( Gaussian( center, _scale_r.getRandDouble(), _std_div_r.getRandDoubleNonLinear(), max_rad ) );
        m_coords.push_back( center );
    }

    int frac = num_gauss /MAX_THREADS__ +1;
    for( size_t it=0; it < MAX_THREADS__; ++it )
    {
        size_t it_sta = it *frac; size_t it_end = (it +1) *frac;
        m_threads.push_back( std::thread( &DataHandler::addSmallGaussianThread, this, max_range, it_sta, it_end ) );
    }

    for( size_t it=0; it < MAX_THREADS__; ++it )
        m_threads[it].join();

    m_threads.clear();
    m_gaussians.clear();
    m_coords.clear();

    std::cout << "Threads joined" << std::endl;
}



inline double lerp( const double val0, const double val1, const double weight )
{
    return val0 + ( val1 -val0 ) *weight;
}

inline double fade( const double val )
{
    return val *val *val *( val *( val *6 -15 ) +10 );
}

template <typename _Tp>
void DataHandler<_Tp>::addPerlinThread( const Range<double>& _scale )
{

    npy::Coordinate voxel( 0, 0, 0 );
    while( true )
    {
        voxel[1] = 0;
        voxel[2] = 0;
        voxel[0] = getNextIter();
        if( voxel[0] >= m_shape[0] )
            break;

        Coordinate grid_cell;
        grid_cell[0] = voxel[0] /m_grid_steps[0];
        CoordinateD grid_pos;
        grid_pos[0] = voxel[0] /m_grid_steps[0] -grid_cell[0];

        size_t mem_it = voxel[0] *m_thread_step;
        do
        {
            grid_cell[1] = voxel[1] /m_grid_steps[1];
            grid_cell[2] = voxel[2] /m_grid_steps[2];

            grid_pos[1] = voxel[1] /m_grid_steps[1] -grid_cell[1];
            grid_pos[2] = voxel[2] /m_grid_steps[2] -grid_cell[2];

            CoordinateD weights = grid_pos;

            weights[0] = fade( grid_pos[0] );
            weights[1] = fade( grid_pos[1] );
            weights[2] = fade( grid_pos[2] );

            double val0, val1, ix0, ix1, iy0, iy1, average;
            val0 = getDistDotGrad( grid_pos, Coordinate( 0,0,0 ), grid_cell );
            val1 = getDistDotGrad( grid_pos, Coordinate( 0,0,1 ), Coordinate( grid_cell[0], grid_cell[1], grid_cell[2]+1 ) );
            ix0 = lerp( val0, val1, weights[2] );
            val0 = getDistDotGrad( grid_pos, Coordinate( 0,1,0 ), Coordinate( grid_cell[0], grid_cell[1]+1, grid_cell[2] )  );
            val1 = getDistDotGrad( grid_pos, Coordinate( 0,1,1 ), Coordinate( grid_cell[0], grid_cell[1]+1, grid_cell[2]+1 )  );
            ix1 = lerp( val0, val1, weights[2] );
            iy0 = lerp( ix0, ix1, weights[1] );

            val0 = getDistDotGrad( grid_pos, Coordinate( 1,0,0 ), Coordinate( grid_cell[0]+1, grid_cell[1], grid_cell[2] ) );
            val1 = getDistDotGrad( grid_pos, Coordinate( 1,0,1 ), Coordinate( grid_cell[0]+1, grid_cell[1], grid_cell[2]+1 ) );
            ix0 = lerp( val0, val1, weights[2] );
            val0 = getDistDotGrad( grid_pos, Coordinate( 1,1,0 ), Coordinate( grid_cell[0]+1, grid_cell[1]+1, grid_cell[2] )  );
            val1 = getDistDotGrad( grid_pos, Coordinate( 1,1,1 ), Coordinate( grid_cell[0]+1, grid_cell[1]+1, grid_cell[2]+1 )  );
            ix1 = lerp( val0, val1, weights[2] );
            iy1 = lerp( ix0, ix1, weights[1] );

            average = lerp( iy0, iy1, weights[0] );

            m_mask[mem_it] = average;

            ++mem_it;
        } while( iterateCoordinate( voxel ) );

    }
}



template <typename _Tp>
void DataHandler<_Tp>::addPerlinNoise( const Coordinate& _num_nodes, const Range<double>& _scale )
{
    Coordinate num_nodes = _num_nodes;
    if( num_nodes[1] == 0 )
        num_nodes[1] = std::round( static_cast<double>( m_shape[1] ) / m_shape[0] * num_nodes[0] );
    if( num_nodes[2] == 0 )
        num_nodes[2] = std::round( static_cast<double>( m_shape[2] ) / m_shape[0] * num_nodes[0] );
    std::cout << "Grid Cells: " << num_nodes << std::endl;

    m_grid_steps[0] = static_cast<double>( m_shape[0] ) / num_nodes[0];
    m_grid_steps[1] = static_cast<double>( m_shape[1] ) / num_nodes[1];
    m_grid_steps[2] = static_cast<double>( m_shape[2] ) / num_nodes[2];

    std::cout << m_grid_steps << std::endl;

    npy::Range<double> grad_d( -1.0, 1.0 );
    m_node_gradients.clear();
    for( size_t it=0; it < num_nodes[0]+1; ++it )
    {
        m_node_gradients.push_back( std::vector< std::vector<CoordinateD>>() );
        for( size_t jt=0; jt < num_nodes[1]+1; ++jt )
        {
            m_node_gradients[it].push_back( std::vector<CoordinateD>() );
            for( size_t kt=0; kt < num_nodes[2]+1; ++kt )
            {
                CoordinateD grad;
                grad[0] = grad_d.getRandDouble();
                grad[1] = grad_d.getRandDouble();
                grad[2] = grad_d.getRandDouble();
                grad.normalize();
                m_node_gradients[it][jt].push_back( grad );
            }
        }
    }
    std::cout << m_node_gradients.size() << ", " << m_node_gradients[0].size() << ", " << m_node_gradients[0][0].size() << std::endl;

    m_mask.resize( m_size );
    m_iter = -1;
    m_thread_step = m_shape[1] *m_shape[2];
    for( size_t it=0; it < g_max_threads; ++it )
        m_threads.push_back( std::thread( &DataHandler::addPerlinThread, this, _scale ) );
    for( size_t it=0; it < g_max_threads; ++it )
        m_threads[it].join();

    _Tp max_elem = std::numeric_limits<_Tp>::min(), min_elem = std::numeric_limits<_Tp>::max();
    for( size_t it=0; it < m_size; ++it )
    {
        if( m_mask[it] > max_elem )
            max_elem = m_mask[it];
        else if ( m_mask[it] < min_elem )
            min_elem = m_mask[it];
    }

    double frac = max_elem - min_elem;
    for( size_t it=0; it < m_size; ++it )
        m_mask[it] = ( m_mask[it] -min_elem ) /frac;

    std::cout << std::endl << min_elem << " > " << max_elem << "->" << frac << std::endl;
    for( size_t it=0; it < m_size; ++it )
        m_data[it] += _scale.getFraction( m_mask[it] );

    m_mask.clear();
    m_threads.clear();
    m_node_gradients.clear();
}



template <typename _Tp>
void DataHandler<_Tp>::addPerVoxelUniformNoise( const Range<double>& _range )
{
    for( size_t it=0; it < m_size; ++it )
    {
        m_data[it] += _range.getRandDouble();
    }
}

template <typename _Tp>
inline _Tp nonLinear( const _Tp _val, const _Tp _diff, const _Tp _offset )
{
    return std::sqrt( ( _val -_offset ) /_diff );
}

template <typename _Tp>
void DataHandler<_Tp>::addGuidedPerVoxelUniformNoise( const Range<double>& _range )
{
    _Tp max_e = std::numeric_limits<_Tp>::min(), min_e = std::numeric_limits<_Tp>::max();
    for( size_t it=0; it < m_size; ++it )
    {
        if( m_data[it] > max_e )
            max_e = m_data[it];
        else if( m_data[it] < min_e )
            min_e = m_data[it];
    }

    _Tp diff = max_e -min_e;
    for( size_t it=0; it < m_size; ++it )
    {
        _Tp noise = _range.getRandDouble() *nonLinear( m_data[it], diff, min_e );
        m_data[it] += noise;
    }
}


}

template class npy::DataHandler<float>;
template class npy::DataHandler<double>;
