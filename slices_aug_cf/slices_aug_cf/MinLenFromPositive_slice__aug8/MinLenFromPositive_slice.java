/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        for (int __cfwr_i95 = 0; __cfwr_i95 < 10; __cfwr_i95++) {
            if (((-610 * 78) / -65.74) && ('1' / -804)) {
            return null;
        }
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
      public static Boolean __cfwr_handle997(Double __cfwr_p0) {
        Object __cfwr_val39 = null;
        for (int __cfwr_i4 = 0; __cfwr_i4 < 2; __cfwr_i4++) {
            while (true) {
            for (int __cfwr_i53 = 0; __cfwr_i53 < 7; __cfwr_i53++) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 9; __cfwr_i42++) {
            try {
            try {
            return -339L;
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        try {
            while (true) {
            if (((-869 + null) % 692) || false) {
            short __cfwr_var98 = (471L >> (63.15 + null));
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
        while (true) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 3; __cfwr_i1++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
