/****************************************************************************************
 *
 * Copyright (c) 2024, Shenyang Institute of Automation, Chinese Academy of Sciences
 *
 * Authors: Yanpeng Jia
 * Contact: jiayanpeng@sia.cn
 *
 ****************************************************************************************/
#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>



template <typename T> inline
void Quaternion2EulerAngle(const T q[4], T ypr[3])
{
	// roll (x-axis rotation)
	T sinr_cosp = T(2) * (q[0] * q[1] + q[2] * q[3]);
	T cosr_cosp = T(1) - T(2) * (q[1] * q[1] + q[2] * q[2]);
	ypr[2] = atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	T sinp = T(2) * (q[0] * q[2] - q[1] * q[3]);
	if (sinp >= T(1))
	{
		ypr[1] = T(M_PI / 2); // use 90 degrees if out of range
	}
	else if (sinp <= T(-1))
	{
		ypr[1] = -T(M_PI / 2); // use 90 degrees if out of range
	}
	else
	{
		ypr[1] = asin(sinp);
	}
	
	// yaw (z-axis rotation)
	T siny_cosp = T(2) * (q[0] * q[3] + q[1] * q[2]);
	T cosy_cosp = T(1) - T(2) * (q[2] * q[2] + q[3] * q[3]);
	ypr[0] = atan2(siny_cosp, cosy_cosp);
};


struct PitchRollFactor
{
	PitchRollFactor(double p, double r, double q_var)
		: p(p), r(r), q_var(q_var) {}

	template <typename T>
	bool operator()(const T* const q_i, T* residuals) const
	{
		T q_i_tmp[4];
		q_i_tmp[0] = q_i[3]; // ceres in w, x, y, z order
		q_i_tmp[1] = q_i[0];
		q_i_tmp[2] = q_i[1];
		q_i_tmp[3] = q_i[2];

		T ypr[3];
		Quaternion2EulerAngle(q_i_tmp, ypr);

		T e[2];
		e[0] = ypr[1] - T(p);
		e[1] = ypr[2] - T(r);

		residuals[0] = T(2) * e[0] / T(q_var);
		residuals[1] = T(2) * e[1] / T(q_var);

		return true;
	}

	static ceres::CostFunction* Create(const double p, const double r, const double q_var) 
	{
		return (new ceres::AutoDiffCostFunction<PitchRollFactor, 2, 4>(new PitchRollFactor(p, r, q_var)));
	}

	double p, r;
	double q_var;
};



struct GroundFactor
{
	GroundFactor(double var, double tz_prev): var(var), tz_prev(tz_prev){}

	template <typename T>
	bool operator()(const T* tz_curr, T* residuals) const
	{
		residuals[0] = (tz_curr[0] - tz_prev) / T(var);

		return true;
	}

	static ceres::CostFunction* Create(const double var, const double tz_prev) 
	{
		return (new ceres::AutoDiffCostFunction<GroundFactor, 1, 1>(new GroundFactor(var, tz_prev)));
	}

	double var;
	double tz_prev;
};















