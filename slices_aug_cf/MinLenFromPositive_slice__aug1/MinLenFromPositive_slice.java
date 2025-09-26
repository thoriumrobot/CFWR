/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class MinLenFromPositive_slice {
  void test(@Positive int x) {
        for (int __cfwr_i93 = 0; __cfwr_i93 < 9; __cfwr_i93++) {
            if (('a' ^ -343L) && ((916L % null) >> (-561 + -713))) {
            if ((
        while (true) {
            try {
            if (((null / true) ^ null) || (null / ('V' + 69.80))) {
            Integer __cfwr_result15 = null;
        }
        } catch (Exception __cfwr_e62) {
            // ignore
        }
            break; // Prevent infinite loops
        }
false - 163) && (null << 39)) {
            for (int __cfwr_i72 = 0; __cfwr_i72 < 4; __cfwr_i72++) {
            while (false) {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 9; __cfwr_i41++) {
            Boolean __cfwr_data44 = null;
        }
            break; // Prevent infinite loops
        }
        }
        }
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
  }

    private float __cfwr_compute640(byte __cfwr_p0, byte __cfwr_p1, Long __cfwr_p2) {
        return (-40.13f ^ -958L);
        if (((null >> null) >> null) || false) {
            if (false || true) {
            for (int __cfwr_i19 = 0; __cfwr_i19 < 2; __cfwr_i19++) {
            return (null << null);
        }
        }
        }
        if (true && false) {
            for (int __cfwr_i86 = 0; __cfwr_i86 < 9; __cfwr_i86++) {
            while ((null - '5')) {
            if (false && (-94.22 * -609)) {
            byte __cfwr_temp91 = ('i' << (717L / null));
        }
            break; // Prevent infinite loops
        }
        }
        }
        while ((96.51 | 558)) {
            while (false) {
            for (int __cfwr_i51 = 0; __cfwr_i51 < 6; __cfwr_i51++) {
            while (false) {
            try {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 9; __cfwr_i14++) {
            if ((618L * (null - null)) || false) {
            return false;
        }
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return -30.47f;
    }
}