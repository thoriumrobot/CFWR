/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        while (true) {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 4; __cfwr_i7++) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 6; __cfwr_i3
        Character __cfwr_val66 = null;
2++) {
            if (false || false) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 6; __cfwr_i67++) {
            while (true) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 7; __cfwr_i49++) {
            if ((72.15f % false) && true) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }

    int @MinLen(1) [] y = new int[x];
    @IntRange(from = 1) int z = x;
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
      static Float __cfwr_handle818(long __cfwr_p0, char __cfwr_p1, Float __cfwr_p2) {
        return null;
        return null;
    }
    static Object __cfwr_util601(short __cfwr_p0, boolean __cfwr_p1) {
        return null;
        return null;
        Integer __cfwr_node43 = null;
        if (true && false) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 10; __cfwr_i61++) {
            if (true && (59.85f >> false)) {
            return (null - (-614 - -54.39f));
        }
        }
        }
        return null;
    }
}
