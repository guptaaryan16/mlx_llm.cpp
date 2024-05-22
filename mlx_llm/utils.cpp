// Utils for mlx_llm.cpp
#pragma once 

#include <iostream>
#include <string>
#include <sstream>

// get_name function for data resolution
template <typename T>
std::string get_name(std::string prelimiter, const T value)
{
    std::ostringstream oss;
    if (prelimiter.empty())
    {
        oss << value;
    }
    else
    {
        oss << prelimiter << "." << value;
    }
    return oss.str();
}

template <typename T>
std::string get_name(std::string &prelimiter, const T value1, const T value2)
{
    std::ostringstream oss;
    if (prelimiter.empty())
    {
        oss << value1 << "." << value2;
    }
    else
    {
        oss << prelimiter << "." << value1 << "." << value2;
    }
    return oss.str();
}

bool endsWith(const std::string &str, const std::string &suffix)
{
    if (suffix.size() > str.size())
        return false;
    return str.substr(str.size() - suffix.size()) == suffix;
}