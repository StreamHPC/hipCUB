// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common_test_header.hpp"

// hipcub API
#include "hipcub/device/device_adjacent_difference.hpp"
#include "hipcub/iterator/counting_input_iterator.hpp"
#include "hipcub/iterator/discard_output_iterator.hpp"
#include "hipcub/iterator/transform_input_iterator.hpp"

#include "test_utils.hpp"
#include "test_utils_data_generation.hpp"

template<class InputIteratorT, class OutputIteratorT, class... Args>
hipError_t dispatch_adjacent_difference(std::true_type /*left*/,
                                        std::true_type /*copy*/,
                                        void*           d_temp_storage,
                                        size_t&         temp_storage_bytes,
                                        InputIteratorT  d_input,
                                        OutputIteratorT d_output,
                                        Args&&... args)
template<class InputIteratorT, class OutputIteratorT, class... Args>
hipError_t dispatch_adjacent_difference(std::true_type /*left*/,
                                        std::true_type /*copy*/,
                                        void*           d_temp_storage,
                                        size_t&         temp_storage_bytes,
                                        InputIteratorT  d_input,
                                        OutputIteratorT d_output,
                                        Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractLeftCopy(d_temp_storage,
                                                                temp_storage_bytes,
                                                                d_input,
                                                                d_output,
                                                                std::forward<Args>(args)...);
    return ::hipcub::DeviceAdjacentDifference::SubtractLeftCopy(d_temp_storage,
                                                                temp_storage_bytes,
                                                                d_input,
                                                                d_output,
                                                                std::forward<Args>(args)...);
}

template<class InputIteratorT, class OutputIteratorT, class... Args>
hipError_t dispatch_adjacent_difference(std::true_type /*left*/,
                                        std::false_type /*copy*/,
                                        void*          d_temp_storage,
                                        size_t&        temp_storage_bytes,
                                        InputIteratorT d_input,
                                        OutputIteratorT /*d_output*/,
                                        Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractLeft(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_input,
                                                            std::forward<Args>(args)...);
    return ::hipcub::DeviceAdjacentDifference::SubtractLeft(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_input,
                                                            std::forward<Args>(args)...);
}

template<class InputIteratorT, class OutputIteratorT, class... Args>
hipError_t dispatch_adjacent_difference(std::false_type /*left*/,
                                        std::true_type /*copy*/,
                                        void*           d_temp_storage,
                                        size_t&         temp_storage_bytes,
                                        InputIteratorT  d_input,
                                        OutputIteratorT d_output,
                                        Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractRightCopy(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_input,
                                                                 d_output,
                                                                 std::forward<Args>(args)...);
    return ::hipcub::DeviceAdjacentDifference::SubtractRightCopy(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_input,
                                                                 d_output,
                                                                 std::forward<Args>(args)...);
}

template<class InputIteratorT, class OutputIteratorT, class... Args>
hipError_t dispatch_adjacent_difference(std::false_type /*left*/,
                                        std::false_type /*copy*/,
                                        void*          d_temp_storage,
                                        size_t&        temp_storage_bytes,
                                        InputIteratorT d_input,
                                        OutputIteratorT /*d_output*/,
                                        Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractRight(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_input,
                                                             std::forward<Args>(args)...);
    return ::hipcub::DeviceAdjacentDifference::SubtractRight(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_input,
                                                             std::forward<Args>(args)...);
}

template<typename Output, typename T, typename BinaryFunction>
template<typename Output, typename T, typename BinaryFunction>
auto get_expected_result(const std::vector<T>& input,
                         const BinaryFunction  op,
                         std::true_type /*left*/)
{
    std::vector<Output> result(input.size());
    std::adjacent_difference(input.cbegin(), input.cend(), result.begin(), op);
    return result;
}

template<typename Output, typename T, typename BinaryFunction>
template<typename Output, typename T, typename BinaryFunction>
auto get_expected_result(const std::vector<T>& input,
                         const BinaryFunction  op,
                         std::false_type /*left*/)
{
    std::vector<Output> result(input.size());
    // "right" adjacent difference is just adjacent difference backwards
    std::adjacent_difference(input.crbegin(), input.crend(), result.rbegin(), op);
    return result;
}

template<class InputT,
         class OutputT  = InputT,
         bool Left      = true,
         bool Copy      = true,
         bool UseGraphs = false>
struct params
{
    using input_type                 = InputT;
    using output_type                = OutputT;
    static constexpr bool left       = Left;
    static constexpr bool copy       = Copy;
    static constexpr bool use_graphs = UseGraphs;
    using input_type                 = InputT;
    using output_type                = OutputT;
    static constexpr bool left       = Left;
    static constexpr bool copy       = Copy;
    static constexpr bool use_graphs = UseGraphs;
};

template<class Params>
class HipcubDeviceAdjacentDifference : public ::testing::Test
{
public:
    using params = Params;
};

typedef ::testing::Types<params<int>,
                         params<int, double>,
                         params<int8_t, int8_t, true, false>,
                         params<float, float, false, true>,
                         params<double, double, true, true>,
                         params<test_utils::half, test_utils::half>,
                         params<test_utils::bfloat16, test_utils::bfloat16>,
                         params<int, int, true, true, true>>
    Params;

TYPED_TEST_SUITE(HipcubDeviceAdjacentDifference, Params);

TYPED_TEST(HipcubDeviceAdjacentDifference, SubtractLeftCopy)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type = typename TestFixture::params::input_type;
    static constexpr std::integral_constant<bool, TestFixture::params::left> left_constant{};
    static constexpr std::integral_constant<bool, TestFixture::params::copy> copy_constant{};
    using output_type
        = std::conditional_t<copy_constant, input_type, typename TestFixture::params::output_type>;
    static constexpr ::hipcub::Difference op;

    hipStream_t stream = 0;
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    using input_type = typename TestFixture::params::input_type;
    static constexpr std::integral_constant<bool, TestFixture::params::left> left_constant{};
    static constexpr std::integral_constant<bool, TestFixture::params::copy> copy_constant{};
    using output_type
        = std::conditional_t<copy_constant, input_type, typename TestFixture::params::output_type>;
    static constexpr ::hipcub::Difference op;

    hipStream_t stream = 0;
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            const auto input = test_utils::get_random_data<input_type>(
                size,
                test_utils::convert_to_device<input_type>(-50),
                test_utils::convert_to_device<input_type>(50),
                seed_value);

            input_type*  d_input{};
            output_type* d_output{};
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(d_input[0])));
            if(copy_constant)
            {
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_output, size * sizeof(d_output[0])));
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            const auto input = test_utils::get_random_data<input_type>(
                size,
                test_utils::convert_to_device<input_type>(-50),
                test_utils::convert_to_device<input_type>(50),
                seed_value);

            input_type*  d_input{};
            output_type* d_output{};
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(d_input[0])));
            if(copy_constant)
            {
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_output, size * sizeof(d_output[0])));
            }
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), size * sizeof(input_type), hipMemcpyHostToDevice));
                hipMemcpy(d_input, input.data(), size * sizeof(input_type), hipMemcpyHostToDevice));

            const auto expected = get_expected_result<output_type>(input, op, left_constant);

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(dispatch_adjacent_difference(left_constant,
                                                   copy_constant,
                                                   nullptr,
                                                   temporary_storage_bytes,
                                                   d_input,
                                                   d_output,
                                                   size,
                                                   op,
                                                   stream));
            const auto expected = get_expected_result<output_type>(input, op, left_constant);

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(dispatch_adjacent_difference(left_constant,
                                                   copy_constant,
                                                   nullptr,
                                                   temporary_storage_bytes,
                                                   d_input,
                                                   d_output,
                                                   size,
                                                   op,
                                                   stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            hipGraph_t graph;
            if(TestFixture::params::use_graphs)
            {
                graph = test_utils::createGraphHelper(stream);
            }

            HIP_CHECK(dispatch_adjacent_difference(left_constant,
                                                   copy_constant,
                                                   d_temporary_storage,
                                                   temporary_storage_bytes,
                                                   d_input,
                                                   d_output,
                                                   size,
                                                   op,
                                                   stream));

            hipGraphExec_t graph_instance;
            if(TestFixture::params::use_graphs)
            {
                graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
            }

            std::vector<output_type> output(size);
            HIP_CHECK(hipMemcpy(output.data(),
                                copy_constant ? d_output : d_input,
                                size * sizeof(output[0]),
                                hipMemcpyDeviceToHost));
            std::vector<output_type> output(size);
            HIP_CHECK(hipMemcpy(output.data(),
                                copy_constant ? d_output : d_input,
                                size * sizeof(output[0]),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_input));
            if(copy_constant)
            {
            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_input));
            if(copy_constant)
            {
                HIP_CHECK(hipFree(d_output));
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(output, expected));

            if(TestFixture::params::use_graphs)
            {
                test_utils::cleanupGraphHelper(graph, graph_instance);
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

// Params for tests
template<bool Left = true, bool Copy = false>
struct DeviceAdjacentDifferenceLargeParams
{
    static constexpr bool left = Left;
    static constexpr bool copy = Copy;
};

template<class Params>
class HipcubDeviceAdjacentDifferenceLargeTests : public ::testing::Test
{
public:
    static constexpr bool left = Params::left;
    static constexpr bool copy = Params::copy;
};

template<unsigned int SamplingRate>
class check_output_iterator
{
public:
    using flag_type = unsigned int;

private:
    class check_output
    {
    public:
        __device__ check_output(flag_type* incorrect_flag, size_t current_index, size_t* counter)
            : current_index_(current_index), incorrect_flag_(incorrect_flag), counter_(counter)
        {}

        __device__
        check_output&
            operator=(size_t value)
        {
            if(value != current_index_)
            {
                //flagflag
                rocprim::detail::atomic_store(incorrect_flag_, 1);
            }
            if(current_index_ % SamplingRate == 0)
            {
                atomicAdd(counter_, 1);
            }
            return *this;
        }

    private:
        size_t     current_index_;
        flag_type* incorrect_flag_;
        size_t*    counter_;
    };

public:
    using value_type        = size_t;
    using reference         = check_output;
    using pointer           = check_output*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;

    __host__ __device__
    check_output_iterator(flag_type* const incorrect_flag, size_t* const    counter)
        : current_index_(0), incorrect_flag_(incorrect_flag), counter_(counter)
    {}

    __device__
    bool operator==(const check_output_iterator& rhs) const
    {
        return current_index_ == rhs.current_index_;
    }
    __device__
    bool operator!=(const check_output_iterator& rhs) const
    {
        return !(*this == rhs);
    }
    __device__
    reference
        operator*()
    {
        return reference(incorrect_flag_, current_index_, counter_);
    }
    __device__
    reference
        operator[](const difference_type distance) const
    {
        return *(*this + distance);
    }
    __device__
    discard_write&
        operator=(T)
    {
        current_index_ += rhs;
        return *this;
    }
};

template<typename T, typename InputIterator, typename UnaryFunction>
HIPCUB_HOST_DEVICE inline auto make_transform_iterator(InputIterator iterator,
                                                       UnaryFunction transform)
{
    return ::hipcub::TransformInputIterator<T, UnaryFunction, InputIterator>(iterator, transform);
}

template<typename T>
struct conversion_op : public std::unary_function<T, discard_write<T>>
{
    HIPCUB_HOST_DEVICE
    auto operator()(const T i) const
    {
        return check_output_iterator(*this) += rhs;
    }
};

template<typename T>
struct flag_expected_op : public std::binary_function<T, T, discard_write<T>>
{
    bool left;
    T    expected;
    T    expected_above_limit;
    int* d_flags;
    flag_expected_op(bool left, T expected, T expected_above_limit, int* d_flags)
        : left(left)
        , expected(expected)
        , expected_above_limit(expected_above_limit)
        , d_flags(d_flags)
    {}

    HIPCUB_HOST_DEVICE
    T operator()(const discard_write<T>& minuend, const discard_write<T>& subtrahend)
    {
        return ++check_output_iterator{*this};
    }
    __host__ __device__
    check_output_iterator
        operator--(int)
    {
        return --check_output_iterator{*this};
    }

private:
    size_t     current_index_;
    flag_type* incorrect_flag_;
    size_t*    counter_;
};

using HipcubDeviceAdjacentDifferenceLargeTestsParams
    = ::testing::Types<DeviceAdjacentDifferenceLargeParams<true, true>,
                       DeviceAdjacentDifferenceLargeParams<false, true>>;

TYPED_TEST_SUITE(HipcubDeviceAdjacentDifferenceLargeTests,
                 HipcubDeviceAdjacentDifferenceLargeTestsParams);

template<class T>
struct discard_write
{
    T value;

    __device__ operator T() const
    {
        return value;
    }
    __device__
    discard_write&
        operator=(T)
    {
        return *this;
    }
};

template<class T, class InputIterator, class UnaryFunction>
HIPCUB_HOST_DEVICE
inline auto make_transform_iterator(InputIterator iterator, UnaryFunction transform)
{
    return ::hipcub::TransformInputIterator<T, UnaryFunction, InputIterator>(iterator, transform);
}

template<class T>
struct conversion_op : public std::unary_function<T, discard_write<T>>
{
    HIPCUB_HOST_DEVICE
    auto operator()(const T i) const
    {
        return discard_write<T>{i};
    }
};

template<class T>
struct flag_expected_op : public std::binary_function<T, T, discard_write<T>>
{
    bool left;
    T    expected;
    T    expected_above_limit;
    int* d_flags;
    flag_expected_op(bool left, T expected, T expected_above_limit, int* d_flags)
        : left(left)
        , expected(expected)
        , expected_above_limit(expected_above_limit)
        , d_flags(d_flags)
    {}

    HIPCUB_HOST_DEVICE
    T operator()(const discard_write<T>& minuend, const discard_write<T>& subtrahend)
    {
        if(left)
        {
            if(minuend == expected && subtrahend == expected - 1)
            {
                d_flags[0] = 1;
            }
            if(minuend == expected_above_limit && subtrahend == expected_above_limit - 1)
            {
                d_flags[1] = 1;
            }
        }
        else
        {
            if(minuend == expected && subtrahend == expected + 1)
            {
                d_flags[0] = 1;
            }
            if(minuend == expected_above_limit && subtrahend == expected_above_limit + 1)
            {
                d_flags[1] = 1;
            }
        }
        return 0;
    }
};

TYPED_TEST(HipcubDeviceAdjacentDifferenceLargeTests, LargeIndicesAndOpOnce)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                    = size_t;
    static constexpr bool left = TestFixture::left;
    static constexpr bool copy = TestFixture::copy;
    static constexpr unsigned int sampling_rate     = 10000;
    using OutputIterator                            = check_output_iterator<sampling_rate>;
    using flag_type                                 = OutputIterator::flag_type;

    SCOPED_TRACE(testing::Message() << "left = " << left << ", copy = " << copy);

    static constexpr hipStream_t stream = 0; // default

    for(std::size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = test_utils::get_large_sizes(seed_value);

        for(const auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            flag_type* d_incorrect_flag;
            size_t*    d_counter;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_incorrect_flag, sizeof(*d_incorrect_flag)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_counter, sizeof(*d_counter)));
            HIP_CHECK(hipMemset(d_incorrect_flag, 0, sizeof(*d_incorrect_flag)));
            HIP_CHECK(hipMemset(d_counter, 0, sizeof(*d_counter)));

            OutputIterator output(d_incorrect_flag, d_counter);

            const auto input = hipcub::CountingInputIterator<T>(T{0});

            // Return the position where the adjacent difference is expected to be written out.
            // When called with consecutive values the left value is returned at the left-handed difference, and the right value otherwise.
            // The return value is coherent with the boundary values.
            const auto op = [](const auto& larger_value, const auto& smaller_value)
            { return (smaller_value + larger_value) / 2 + (left ? 1 : 0); };

            static constexpr auto left_tag = std::integral_constant<bool, left>{};

            static constexpr auto copy_tag = std::integral_constant<bool, copy>{};

            // Allocate temporary storage
            std::size_t temp_storage_size = 0;
            void*       d_temp_storage    = nullptr;
            HIP_CHECK(dispatch_adjacent_difference(left_tag,
                                                   copy_tag,
                                                   d_temp_storage,
                                                   temp_storage_size,
                                                   input,
                                                   output,
                                                   size,
                                                   op,
                                                   stream));

#ifdef __HIP_PLATFORM_AMD__
            ASSERT_GT(temp_storage_size, 0U);
#endif
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size));

            HIP_CHECK(hipMemsetAsync(d_incorrect_flag, 0, sizeof(*d_incorrect_flag), stream));
            HIP_CHECK(hipMemsetAsync(d_counter, 0, sizeof(*d_counter), stream));

            // Run
            HIP_CHECK(dispatch_adjacent_difference(left_tag,
                                                   copy_tag,
                                                   d_temp_storage,
                                                   temp_storage_size,
                                                   input,
                                                   output,
                                                   size,
                                                   op,
                                                   stream));

            // Copy output to host
            flag_type incorrect_flag;
            size_t    counter;
            HIP_CHECK(hipMemcpy(&incorrect_flag,
                                d_incorrect_flag,
                                sizeof(incorrect_flag),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&counter, d_counter, sizeof(counter), hipMemcpyDeviceToHost));

            ASSERT_EQ(flags[0], 1);
            ASSERT_EQ(flags[1], 1);
            HIP_CHECK(hipFree(d_temp_storage));
            HIP_CHECK(hipFree(d_incorrect_flag));
            HIP_CHECK(hipFree(d_counter));
        }
    }
}