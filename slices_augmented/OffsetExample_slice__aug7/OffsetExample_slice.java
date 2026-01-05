/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class OffsetExample_slice {
    @Positive
  void example2(int @MinLen(2) [] a) {
        return null;

    @Positive
    int j = 2;
    @Positive
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

    protected static char __cfwr_util153(Object __cfwr_p0) {
        Long __cfwr_entry83 = null;
        while (((92.74 & null) | (null / 833))) {
            while ((false >> null)) {
            long __cfwr_entry56 = (false - null);
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        try {
            return null;
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        Float __cfwr_var65 = null;
        return 'A';
    }
    Integer __cfwr_aux751(Object __cfwr_p0) {
        for (int __cfwr_i28 = 0; __cfwr_i28 < 10; __cfwr_i28++) {
            if (((26.48f + null) | 58.76f) || true) {
            while (false) {
            return -75;
            break; // Prevent infinite loops
        }
        }
        }
        Boolean __cfwr_data71 = null;
        try {
            try {
            Boolean __cfwr_result52 = null;
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        String __cfwr_var44 = "result41";
        return null;
    }
    public byte __cfwr_calc200() {
        try {
            if (true && ((null / 7.42) + 43.29)) {
            if (true && ('K' / null)) {
            return null;
        }
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        while (false) {
            try {
            double __cfwr_item77 = (-86.41 >> (false + 'Q'));
        } catch (Exception __cfwr_e71) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}