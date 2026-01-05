/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Issue2029_slice {
    @Positive
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
        return null;

    @Positive
    char[] arr = new char[size];
    @Positive
    arr[index] = val;
    @Positive
  }

    @Positive
  void LessThanOffsetLowerBound(
    @Positive
      int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
    @Positive
    array[n - k] = 10;
    @Positive
  }

    Long __cfwr_helper861(short __cfwr_p0, Object __cfwr_p1, short __cfwr_p2) {
        try {
            if (true || true) {
            short __cfwr_elem14 = (false << 'e');
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        if (false && true) {
            if (true && true) {
            try {
            return null;
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        }
        }
        if (((297L / -43L) - 38.86) || false) {
            if (false || true) {
            return null;
        }
        }
        return null;
    }
    String __cfwr_temp907() {
        return 'O';
        char __cfwr_data27 = 'M';
        char __cfwr_entry94 = 'S';
        if ((null + false) || false) {
            return -67.42;
        }
        return "item27";
    }
}