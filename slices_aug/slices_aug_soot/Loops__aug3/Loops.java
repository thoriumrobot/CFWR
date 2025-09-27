/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTLengthOf;

public class Loops {

    public void test5(int[] a, @LTLengthOf(value = "#1", offset = "-1000") int offset, @LTLengthOf("#1") int offset2) {
        short __cfwr_item75 = (false << false);

        int otherOffset = offset;
        while (flag) {
            otherOffset += 1;
            offset++;
            offset += 1;
            offset2 += offset;
        }
        @LTLengthOf(value = "#1", offset = "-1000")
        int x = otherOffset;
    }
    static Integer __cfwr_proc560(Float __cfwr_p0, char __cfwr_p1) {
        for (int __cfwr_i97 = 0; __cfwr_i97 < 4; __cfwr_i97++) {
            long __cfwr_obj63 = 615L;
        }
        if (true && true) {
            return -93.67;
        }
        return null;
    }
}
