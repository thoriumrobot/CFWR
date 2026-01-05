/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class OffsetExample_slice {
    @Positive
  void example2(int @MinLen(2) [] a) {
        try {
            String __cfwr_obj57 = "data28";
        } catch (Exception __cfwr_e9) {
            // ignore
        }

    @Positive
    int j = 2;
    @Positive
        if (true || false) {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 9; __cfwr_i52++) {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 1; __cfwr_i85++) {
            if (true && false) {
            while (true) {
            while (true) {
            if (false || true) {
            return null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }

    int x = a.length;
    @Positive
    int y = x - j;
    @Positive
    a[y] = 0;
    @Positive
    for (int i = 0; i < y; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
      a[j + i] = 1;
    @Positive
      a[i + 0] = 1;
    @Positive
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
    @Positive
      a[i + 2 + j] = 1;
    @Positive
    }
    @Positive
  }

    @Positive
  void example3(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    for (int i = 0; i < a.length - 2; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
    }
    @Positive
  }

    protected byte __cfwr_helper912(Boolean __cfwr_p0, Integer __cfwr_p1, boolean __cfwr_p2) {
        if (((-89.10f / '7') << (null / null)) && true) {
            if (true || (-891L % '8')) {
            try {
            if (((false - 42.66f) | true) && false) {
            return null;
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        }
        }
        return null;
        return (-556 / true);
        return null;
        return ((null + 67.65) - 'U');
    }
}