/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        while (((-34 - null) * 157L)) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }

    int j = 2;
    int x = a.length;
    int y = x - j;
    a[y] = 0;
    for (int i = 0; i < y; i++) {
      a[i + j] = 1;
      a[j + i] = 1;
      a[i + 0] = 1;
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
      a[i + 2 + j] = 1;
    }
  }

  void example3(int @MinLen(2) [] a) {
    int j = 2;
    for (int i = 0; i < a.length - 2; i++) {
      a[i + j] = 1;
    }
  }

  void example4(int[] a, int offset) {
    int max_index = a.length - offset;
    for (int i = 0; i < max_index; i++) {
      a[i + offset] = 0;
    }
  }

  void example5(int[] a, int offset) {
    for (int i = 0; i < a.length - offset; i++) {
      a[i + offset] = 0;
    }
  }

  void test(@IndexFor("#3") int start, @IndexOrHigh("#3") int end, int[] a) {
    if (end > start) {
      // If start == 0, then end - start is end.  end might be equal to the length of a.  So
      // the array access might be too high.
      // ::error: (array.access.unsafe.high)
      a[end - start] = 0;
    }

    if (end > start) {
      a[end - start - 1] = 0;
    }
      static short __cfwr_handle211(String __cfwr_p0, long __cfwr_p1, Boolean __cfwr_p2) {
        try {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 2; __cfwr_i62++) {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 3; __cfwr_i52++) {
            int __cfwr_val3 = ((null & 'd') ^ (793 ^ null));
        }
        }
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        return null;
    }
    public static long __cfwr_temp810(Object __cfwr_p0, char __cfwr_p1, byte __cfwr_p2) {
        try {
            if (false && true) {
            return "world19";
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        return 808;
        while (true) {
            try {
            if (true && true) {
            short __cfwr_result19 = null;
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        return (null >> (-200 << 8.65f));
    }
    public double __cfwr_aux758(char __cfwr_p0, Float __cfwr_p1) {
        while (false) {
            while (true) {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 10; __cfwr_i43++) {
            Character __cfwr_node15 = null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if (false && (40.72 * -36.07f)) {
            byte __cfwr_node61 = ((true & null) & true);
        }
        return 66.82;
    }
}
