#pragma once

#define TABLE_SIZE 512

static short m_sigmoid_table[TABLE_SIZE];
static short m_tanh_table[TABLE_SIZE];

template <class BaseType, size_t FracDigits> 
class FixedPoint
{
private:
	static const BaseType scaling_factor = (BaseType)(1LL << (BaseType)FracDigits);
	static const BaseType step_factor = (BaseType)((1LL << 16) / TABLE_SIZE);

	bool initd = false;

	void InitTables()
	{
		//double minX = -4.0f;
		double x, y;

		// Sigmoid table, store positive values
		for (unsigned int i = 0; i < TABLE_SIZE/2; ++i)
		{
			x = (double)i * (double)step_factor / (double)scaling_factor;
			y = 1.0 / (1.0 + exp(-x));
			m_sigmoid_table[i] = (BaseType)(y * scaling_factor);
			//std::cout << "sigm(" << x << ") = " << y << ", m_sigmoid_table[" << i << "] = " << m_sigmoid_table[i] << std::endl;
		}

		// Sigmoid table, store negative values
		for (unsigned int i = TABLE_SIZE/2; i < TABLE_SIZE; ++i)
		{
			x = -(double)(TABLE_SIZE - i) * (double)step_factor / (double)scaling_factor;
			y = 1.0 / (1.0 + exp(-x));
			m_sigmoid_table[i] = (BaseType)(y * scaling_factor);
			//std::cout << "sigm(" << x << ") = " << y << ", m_sigmoid_table[" << i << "] = " << m_sigmoid_table[i] << std::endl;
		}

		// Tanh table, store positive values
		for (unsigned int i = 0; i < TABLE_SIZE/2; ++i)
		{
			x = (double)i * (double)step_factor / (double)scaling_factor;
			y = tanh(x);
			m_tanh_table[i] = (BaseType)(y * scaling_factor);
          	//std::cout << "tanh(" << x << ") = " << y << ", m_tanh_table[" << i << "] = " << m_tanh_table[i] << std::endl;
		}

		// Tanh table, store negative values
		for (unsigned int i = TABLE_SIZE/2; i < TABLE_SIZE; ++i)
		{
			x = -(double)(TABLE_SIZE - i) * (double)step_factor / (double)scaling_factor;
			y = tanh(x);
			m_tanh_table[i] = (BaseType)(y * scaling_factor);
          	//std::cout << "tanh(" << x << ") = " << y << ", m_tanh_table[" << i << "] = " << m_tanh_table[i] << std::endl;
		}
	}

public:

	FixedPoint()
	{
		m_value = 0;
	}

	FixedPoint(float initValue)
	{
		m_value = (BaseType)(initValue * scaling_factor);
	}

	FixedPoint(double initValue)
	{
		m_value = (BaseType)(initValue * scaling_factor);
	}

	FixedPoint(BaseType baseValue)
	{
		m_value = baseValue;
	}

	void operator=(const FixedPoint &d)
	{
		m_value = d.m_value;
	}

	void operator=(const float val)
	{
		m_value = (BaseType)(val * scaling_factor);
	}

	void operator=(const double val)
	{
		m_value = (BaseType)(val * scaling_factor);
	}

	static size_t numBits()
	{
		return sizeof(BaseType) * 8;
	}

	static size_t numIntBits()
	{
		return sizeof(BaseType)*8 - FracDigits;
	}

	static size_t numFracBits()
	{
		return FracDigits;
	}

	FixedPoint sigmoid()
	{
		// Hack for now
		//double floatValue = (double)m_value / scaling_factor;
		//double sigm = 1.0 / (1.0 + exp(-floatValue));
		//return FixedPoint(sigm);

		if (!initd)
		{
			InitTables();
			initd = true;
		}
		unsigned short idx = (unsigned short)m_value / step_factor;
		return m_sigmoid_table[idx];

#if 0
		if (!initd)
		{
			InitTables();
			initd = true;
		}

		long long integer = (long long)m_value / scaling_factor;

		if (integer <= -4)
		{
			return FixedPoint(0.0f);
		}
		else if (integer >= 4)
		{
			return FixedPoint(1.0f);
		}
		else
		{
			FixedPoint rescaled = (FixedPoint(m_value) + 4.0) * 16.0;
			BaseType addr = (rescaled.m_value / scaling_factor) % 128;
			return FixedPoint(m_sigmoid_table[addr]); //FixedPoint(1.0 / (1.0 + exp(-m_value)));
		}
#endif
	}

	FixedPoint hypertan()
	{
		// Hack for now
		//double floatValue = (double)m_value / scaling_factor;
		//double result = tanh(floatValue);
		//return FixedPoint(result);

		if (!initd)
		{
			InitTables();
			initd = true;
		}
		unsigned short idx = (unsigned short)m_value / step_factor;
        return m_tanh_table[idx];
	}

	double toDouble()
	{
		double doubleValue = (double)m_value / scaling_factor;
		return doubleValue;
	}

	float toFloat()
	{
		float floatValue = (float)m_value / scaling_factor;
		return floatValue;
	}
	

	friend FixedPoint operator+(const FixedPoint &left, const FixedPoint &right)
	{
		long long total = (long long)left.m_value + (long long)right.m_value;
		BaseType scaled = (BaseType)total;
		return FixedPoint(scaled);
	}

	friend FixedPoint operator*(const FixedPoint &left, const FixedPoint &right)
	{
		DWORD32 product = (DWORD32)left.m_value * (DWORD32)right.m_value;

		product = product >> FracDigits;

		BaseType scaled = (BaseType)product;
		return FixedPoint(scaled);
	}

	friend FixedPoint four_input_mult_add(FixedPoint &a0, FixedPoint &b0, FixedPoint &a1, FixedPoint &b1)
	{
		DWORD32 p0 = (DWORD32)a0.m_value * (DWORD32)b0.m_value;
		DWORD32 p1 = (DWORD32)a1.m_value * (DWORD32)b1.m_value;
		DWORD sum = p0 + p1;

		sum = sum >> FracDigits;

		BaseType scaled = (BaseType)sum;
		return FixedPoint(scaled);
	}

	friend FixedPoint eight_input_mult_add(FixedPoint &a0, FixedPoint &b0, FixedPoint &a1, FixedPoint &b1, FixedPoint &a2, FixedPoint &b2, FixedPoint &a3, FixedPoint &b3)
	{
		DWORD32 p0 = (DWORD32)a0.m_value * (DWORD32)b0.m_value;
		DWORD32 p1 = (DWORD32)a1.m_value * (DWORD32)b1.m_value;
		DWORD32 p2 = (DWORD32)a2.m_value * (DWORD32)b2.m_value;
		DWORD32 p3 = (DWORD32)a3.m_value * (DWORD32)b3.m_value;
		DWORD sum = p0 + p1 + p2 + p3;

		sum = sum >> FracDigits;

		BaseType scaled = (BaseType)sum;
		return FixedPoint(scaled);
	}

	friend bool operator<(const FixedPoint &left, const FixedPoint &right)
	{
		return left.m_value < right.m_value;
	}

	friend bool operator>(const FixedPoint &left, const FixedPoint &right)
	{
		return left.m_value > right.m_value;
	}

	FixedPoint operator+=(const FixedPoint &right)
	{
		m_value += right.m_value;
		return FixedPoint(m_value);
	}

	friend FixedPoint operator-(const FixedPoint &left, const FixedPoint &right)
	{
		long long total = (long long)left.m_value - (long long)right.m_value;
		BaseType scaled = (BaseType)total;
		return FixedPoint(scaled);
	}

	friend std::ostream& operator<<(std::ostream& os, const FixedPoint &elem) {
		double scalar = (double)elem.m_value / scaling_factor;
		return os << scalar;
	}

	BaseType m_value;
};