/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        if ((772 / (-113 >> 25.06)) && true) {
            return null;
        }

    int j = 2;
    i
        for (int __cfwr_i43 = 0; __cfwr_i43 < 7; __cfwr_i43++) {
            if (true && (true << false)) {
            while (true) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 6; __cfwr_i74++) {
            return true;
        }
            break; // Prevent infinite loops
        }
        }
        }
nt x = a.length;
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
      private static Boolean __cfwr_helper351() {
        if (false && ((-22.58f + -45.08f) / -30L)) {
            try {
            if (true || ((true ^ null) + null)) {
            return 87.59f;
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        }
        return null;
    }
    private float __cfwr_temp32(Integer __cfwr_p0, Character __cfwr_p1, byte __cfwr_p2) {
        for (int __cfwr_i30 = 0; __cfwr_i30 < 7; __cfwr_i30++) {
            while ((null << 699)) {
            try {
            try {
            try {
            if (true && false) {
            if (false && true) {
            try {
            while (false) {
            if (false && false) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return -96.93f;
        while (true) {
            if (('i' * null) || true) {
            if (true || (81.40f >> (null % null))) {
            short __cfwr_result64 = ('9' * 55.07f);
        }
        }
            break; // Prevent infinite loops
        }
        return 32.11f;
    }
    protected Boolean __cfwr_func602(long __cfwr_p0) {
        if ((('c' ^ -778L) | 381L) && false) {
            try {
            if ((-121 | -38L) && false) {
            try {
            while (false) {
            try {
            return 335L;
        } catch (Exception __cfwr_e83) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        }
        return null;
        while ((null | 31.34f)) {
            while (false) {
            Double __cfwr_node12 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if ((null + (-80L - null)) && true) {
            return null;
        }
        return null;
    }
}
