/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanCustomCollection_slice {
    @Positive
  private LessThanCustomCollection(int[] array) {
        while (false) {
            if (false && (84.31 ^ 'o')) {
            if ((233L << (31.55f ^ true)) || false) {
            for (int __cfwr_i27 = 0; __cfwr_i27 < 7; __cfwr_i27++) {
            for (int __cfwr_i21 = 0; __c
        if (false && (176 & (95.71f << -354))) {
            try {
            for (int __cfwr_i98 = 0; __cfwr_i98 < 2; __cfwr_i98++) {
            if ((null * ('b' * false)) && true) {
            Integer __cfwr_var11 = null;
        }
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        }
fwr_i21 < 1; __cfwr_i21++) {
            return null;
        }
        }
        }
        }
            break; // Prevent infinite loops
        }

    @Positive
    this(array, 0, array.length);
    @Positive
  }

    @Positive
  private LessThanCustomCollection(
    @Positive
      int[] array, @IndexOrHigh("#1") @LessThan("#3 + 1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    this.array = array;
    // can't est. that end - start is the length of this.
    // :: error: (assignment)
    @Positive
    this.end = end;
    // start is @LessThan(end + 1) but should be @LessThan(this.end + 1)
    // :: error: (assignment)
    @Positive
    this.start = start;
    @Positive
  }

    @Positive
  public @LengthOf("this") int length() {
    @Positive
    return end - start;
    @Positive
  }

    @Positive
  public double get(@IndexFor("this") int index) {
    // TODO: This is a bug.
    // :: error: (argument)
    @Positive
    checkElementIndex(index, length());
    // Because index is an index for "this" the index + start
    // must be an index for array.
    // :: error: (array.access.unsafe.high)
    @Positive
    return array[start + index];
    @Positive
  }

    private String __cfwr_temp867(Double __cfwr_p0, byte __cfwr_p1) {
        return null;
        if (true || (104 & null)) {
            int __cfwr_item87 = -96;
        }
        if (true && false) {
            while (true) {
            try {
            try {
            try {
            Double __cfwr_node18 = null;
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        Boolean __cfwr_val73 = null;
        return "data9";
    }
    byte __cfwr_compute553() {
        Character __cfwr_entry7 = null;
        return 198;
        return "temp64";
        return (-602L * null);
    }
    private static char __cfwr_util25(Long __cfwr_p0) {
        try {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 3; __cfwr_i67++) {
            return null;
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        return 'x';
    }
}