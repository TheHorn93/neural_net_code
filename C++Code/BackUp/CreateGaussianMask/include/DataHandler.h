#ifndef NPY_DATA_HANDLER_H_
#define NPY_DATA_HANDLER_H_

#include "Utils.h"
#include "FileHandler.h"
#include <vector>
#include <cmath>
#include <thread>
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

    void addMultipleGaussians( const Range<size_t> num_gauss,
                               const Range<double> scale_r,
                               const Range<double> std_div_r );

private:
    inline bool CoordInList( const Coordinate& _new_pos )
    {
        for( const Coordinate& pos : m_coords )
            if( pos == _new_pos )
                return true;
        return false;
    }

    std::unique_ptr<npy::FileHandler<_Tp>> m_f_handler;
    npy::Coordinate m_shape;
    size_t m_size;
    _Tp* m_data;
    std::vector<npy::Gaussian> m_gaussians;
    std::vector<npy::Coordinate> m_coords;
};

}

#endif // NPY_DATA_HANDLER_H_
