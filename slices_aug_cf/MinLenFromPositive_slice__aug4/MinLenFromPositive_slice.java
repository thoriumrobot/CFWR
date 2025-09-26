/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        while ((('w' | -48.73) / (null - 'n'))) {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 5; __cfwr_i7++) {
            for (int __cfwr_i99 = 0
        byte __cfwr_temp85 = ('E' | null);
; __cfwr_i99 < 2; __cfwr_i99++) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 1; __cfwr_i90++) {
            while (true) {
            for (int __cfwr_i60 = 0; __cfwr_i60 < 3; __cfwr_i60++) {
            return null;
        }
            break; // Prevent infinite loops
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
      protected static Double __cfwr_temp862(Boolean __cfwr_p0) {
        if (false && ((93.34f * null) >> -942)) {
            try {
            if ((-258L % (false << 87L)) && false) {
            while ((7.10f + 351)) {
            while (false) {
            if (false || ((false % 97.71) ^ 11.78)) {
            while (false) {
            while (true) {
            if (((true - 31.20f) * (null * 't')) || true) {
            while (true) {
            double __cfwr_entry89 = (true << ('9' / false));
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        }
        if (true || true) {
            while (('l' % null)) {
            if (('r' << 74.97) || ((85.91f - true) << (32.74 ^ 'f'))) {
            Float __cfwr_entry10 = null;
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}
