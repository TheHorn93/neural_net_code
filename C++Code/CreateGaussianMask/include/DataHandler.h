#ifndef NPY_DATA_HANDLER_H_
#define NPY_DATA_HANDLER_H_

#include "Utils.h"
#include "FileHandler.h"
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <Volume.h>
//#include <opencv2/opencv.hpp>

namespace npy
{

template <typename _Tp>
class DataHandler
{
public:
    DataHandler( const std::string& i_file_path, const std::string& o_file_path, npy::Coordinate shape );
    DataHandler( const std::string& o_file_path, npy::Coordinate shape );
    ~DataHandler();

    void addMultipleGaussians( const Range<size_t>& num_gauss,
                               const Range<double>& scale_r,
                               const Range<double>& std_div_r );

    void addStructuredGaussians( const Range<size_t>& num_structures,
                                 const Range<double>& str_std_div,
                                 const Range<size_t>& num_gauss,
                                 const Range<double>& scale_r,
                                 const Range<double>& std_div_r );

    void addSmallGaussians( const Range<size_t>& num_gauss,
                            const Range<double>& scale_r,
                            const Range<double>& std_div_r );

    void addPerVoxelUniformNoise( const Range<double>& range );

    void addGuidedPerVoxelUniformNoise( const Range<double>& range );

    void addPerlinNoise( const Coordinate& num_nodes, const Range<double>& scale );

    void clip( const double lower_val, const double upper_val );

private:
    void evalGauss( const size_t max_div, const double iter );
    void addGaussianThread( const size_t max_range, const double g_idx );
    void createStructureGaussians( const npy::Range<size_t>& num_gauss, const npy::Range<double>& std_div_r, std::vector<npy::Gaussian>& output );
    void addPerlinThread( const Range<double>& scale );
    void addSmallGaussianThread( const size_t max_range, const size_t it_st, const size_t it_end );

    inline double getDistDotGrad( const CoordinateD& _grid_pos, const Coordinate& _grid_node, const Coordinate& _node ) const
    {
        CoordinateD dist = _grid_pos;
        dist = dist - _grid_node;
        //dist.normalize();
        return dist.dotProduct( _node.iter( m_node_gradients ) );
    }

    inline bool CoordInList( const Coordinate& _new_pos )
    {
        for( const Coordinate& pos : m_coords )
            if( pos == _new_pos )
                return true;
        return false;
    }

    inline bool iterateCoordinate( npy::Coordinate& _iter )
    {
        if( _iter[2] < m_shape[2] -1 )
            ++_iter[2];
        else
        {
            _iter[2] = 0;
            if( _iter[1] < m_shape[1] -1 )
                ++_iter[1];
            else
                return false;
        }
        return true;
    }

    inline int getNextIter()
    {
        std::unique_lock<std::mutex> iter_lock( m_iter_mutex );
        ++m_iter;
        std::cout << "\rThread " << std::this_thread::get_id() << ": Got iter " << m_iter << " of " << m_shape[0];
        return m_iter;
    }

    inline double getStrStdDiv( const npy::Coordinate& _center, const npy::Range<double>& _std_div_r, const std::vector<npy::Gaussian>& _str_element )
    {
        double max_elem = 0.0;
        for( size_t it=0; it < _str_element.size(); ++it )
        {
            double expo = std::pow( _center.getSqrdDistance( _str_element[it].center ), 2 ) / ( 2 *_str_element[it].sqrt_rad );
            double eval = std::exp( -expo );
            if( eval > max_elem )
                max_elem = eval;
        }
        return _std_div_r.getFraction( max_elem );
    }

    std::unique_ptr<npy::FileHandler<_Tp>> m_f_handler;
    npy::Coordinate m_shape;
    size_t m_size;
    Volume<_Tp> m_data;
    std::vector<_Tp> m_mask;

    std::vector<npy::Gaussian> m_gaussians;
    std::vector<npy::Coordinate> m_coords;
    std::vector<double> m_gauss_vals;
    std::vector<std::vector<std::vector<npy::CoordinateD>>> m_node_gradients;
    npy::CoordinateD m_grid_steps;

    std::vector<std::thread> m_threads;
    std::mutex m_iter_mutex;
    int m_iter;
    size_t m_thread_step;
};

}

#endif // NPY_DATA_HANDLER_H_
