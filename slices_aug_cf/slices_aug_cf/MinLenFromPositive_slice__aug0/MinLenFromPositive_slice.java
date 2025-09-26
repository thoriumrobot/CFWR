/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        for (int __cfwr_i88 = 0; __cfwr_i88 < 6; __cfwr_i88++) {
            return -141L;
        }

    int @MinLen(1) [] y = new int[x];
    @IntRan
        for (int __cfwr_i69 = 0; __cfwr_i69 < 8; __cfwr_i69++) {
            try {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 4; __cfwr_i15++) {
            try {
            return (null ^ -598);
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        }
ge(from = 1) int z = x;
    @Positive int q = x;
  }

  @SuppressWarnings("index")
  void foo(int x) {
    test(x);
  }

  void foo2(int x) {
    // :: error: (argument)
    test(x);
  }

  void test_lub1(boolean flag, @Positive int x, @IntRange(from = 6, to = 25) int y) {
    int z;
    if (flag) {
      z = x;
    } else {
      z = y;
    }
    @Positive int q = z;
    @IntRange(from = 1) int w = z;
  }

  void test_lub2(boolean flag, @Positive int x, @IntRange(from = -1, to = 11) int y) {
    int z;
    if (flag) {
      z = x;
    } else {
      z = y;
    }
    // :: error: (assignment)
    @Positive int q = z;
    @IntRange(from = -1) int w = z;
  }

  @Positive int id(@Positive int x) {
    return x;
  }

  void test_id(int param) {
    @Positive int x = id(5);
    @IntRange(from = 1) int y = id(5);

    int @MinLen(1) [] a = new int[id(100)];
    // :: error: (assignment)
    int @MinLen(10) [] c = new int[id(100)];

    int q = id(10);

    if (param == q) {
      int @MinLen(1) [] d = new int[param];
    }
      public short __cfwr_func446() {
        while (true) {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 3; __cfwr_i63++) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 5; __cfwr_i90++) {
            Float __cfwr_elem50 = null;
        }
        }
            break; // Prevent infinite loops
        }
        try {
            while (true) {
            Float __cfwr_result64 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        return (-95.16f << true);
        try {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 7; __cfwr_i14++) {
            return 'L';
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        return null;
    }
    protected boolean __cfwr_calc144(Character __cfwr_p0, byte __cfwr_p1, short __cfwr_p2) {
        if (true && false) {
            while (true) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 5; __cfwr_i1++) {
            Boolean __cfwr_result34 = null;
        }
            break; // Prevent infinite loops
        }
        }
        return false;
    }
    public static Object __cfwr_calc866(Integer __cfwr_p0, float __cfwr_p1, Float __cfwr_p2) {
        Character __cfwr_temp10 = null;
        if (true || false) {
            Integer __cfwr_val81 = null;
        }
        return null;
    }
}
