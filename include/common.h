#ifndef SAMPLE_COMMON_INTERFACE_HPP_
#define SAMPLE_COMMON_INTERFACE_HPP_

#include <thread>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <fstream>

#define TIME_MEASUREMENT 1

static int BoundCPU(const pthread_t& thread_id, int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    int ret = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), &cpuset);
    if(ret != 0) {
        std::cerr << "cpu bound failed." << std::endl;
    }
    return ret;
}

class SampleTimer{
public:

#ifdef TIME_MEASUREMENT
    SampleTimer(const std::string& name):
        name_(name)

    {
        memset(&timeBegin_, 0, sizeof(struct timeval));
        memset(&timeBegin_, 0, sizeof(struct timeval));

        gettimeofday(&timeBegin_, NULL);
        LOG_INFO("timer start ...");
    }

    ~SampleTimer()
    {
        gettimeofday(&timeEnd_, NULL);
        int msec = (timeEnd_.tv_sec - timeBegin_.tv_sec) * 1000 + (timeEnd_.tv_usec - timeBegin_.tv_usec) / 1000;
        LOG_INFO(name_.c_str() << " Used Time: " << msec << " ms");
    }
#else
    SampleTimer(const std::string& funcName){}
    ~SampleTimer(){}
#endif

private:
    SampleTimer(){}
    SampleTimer& operator=(const SampleTimer& obj)
    {
        return *this;
    }

private:
    std::string name_;
    struct timeval timeBegin_;
    struct timeval timeEnd_;
};

int getMMZRemainSize() {
	std::string filename = "/proc/media-mem";
	std::ifstream fin;
	fin.open(filename);
	if(!fin.is_open()) {
		LOG_ERROR("open " << filename << " failed!");
		return 0;
	}
	std::string cur_line;
	std::string last_line;

	while (!fin.eof()) {
		getline(fin, cur_line);
		if(cur_line.length() > 0) {
			last_line = cur_line;
		}
	}

	fin.close();

	if(last_line.empty()) {
		LOG_ERROR("read mmz remain info failed!");
		return 0;
	}

	std::string mem_info_str = last_line;

	mem_info_str = mem_info_str.substr(mem_info_str.find("remain="));

	size_t sub_size = mem_info_str.find_first_of("KB") - mem_info_str.find_first_of("=") - 1;
	if(sub_size > 0 && sub_size < mem_info_str.length()) {
		mem_info_str = mem_info_str.substr(mem_info_str.find_first_of("=") + 1, sub_size);
	}

	int cur_mmz_remain = 0;
	try {
		cur_mmz_remain = std::stoi(mem_info_str);
	} catch(const std::invalid_argument &e) {
		LOG_ERROR("invalid argument : " << e.what());
	}
	return cur_mmz_remain;
}


#endif