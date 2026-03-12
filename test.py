def sqrt_positive_int(n):
	"""Return the square root of a positive integer `n` as a float.

	Raises ValueError if `n` is not an int or not positive.
	"""
	import math
	if not isinstance(n, int):
		raise ValueError("input must be an integer")
	if n <= 0:
		raise ValueError("input must be a positive integer")
	return math.sqrt(n)


if __name__ == "__main__":
	while True:
		s = input("Enter a positive integer (or 'q' to quit): ")
		if s.lower() == 'q':
			print("Exiting.")
			break
		try:
			n = int(s)
			print("Square root:", sqrt_positive_int(n))
		except ValueError as e:
			print("Invalid input:", e)
