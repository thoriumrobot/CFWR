/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        if ((-18.00f / -7.97f) && true) {
            while ((-39.62f % null)) {
            if (true || true) {
            return 149;
        }
    
        if (false || false) {
            while ((-92.51f | (null * 's'))) {
            while (true) {
            try {
            return null;
        } catch (Exception __cfwr_e11) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        break; // Prevent infinite loops
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
      double __cfwr_compute591(Float __cfwr_p0, long __cfwr_p1) {
        while (false) {
            Object __cfwr_node47 = null;
            break; // Prevent infinite loops
        }
        if ((null ^ true) && (true - (486L ^ true))) {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 8; __cfwr_i43++) {
            if (true && true) {
            return ('o' >> ('2' ^ 411));
        }
        }
        }
        return 21.23;
    }
    static Boolean __cfwr_temp369(byte __cfwr_p0) {
        if ((null << (null / null)) && ((90.86f % null) + (null | -96.89f))) {
            while (((null ^ 450) >> 401L)) {
            return true;
            break; // Prevent infinite loops
        }
        }
        if (false && true) {
            try {
            if (true || (null / 16.19f)) {
            byte __cfwr_val54 = (true + null);
        }
        } catch (Exception __cfwr_e87) {
            // ignore
        }
        }
        Character __cfwr_obj16 = null;
        try {
            return -885L;
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        return null;
    }
    public Integer __cfwr_temp406(byte __cfwr_p0) {
        if (true && false) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 2; __cfwr_i14++) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        }
        }
        int __cfwr_entry36 = -408;
        while ((null / ('H' * false))) {
            if (false && false) {
            try {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 4; __cfwr_i11++) {
            return null;
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        if (true || false) {
            return null;
        }
        return null;
    }
}
