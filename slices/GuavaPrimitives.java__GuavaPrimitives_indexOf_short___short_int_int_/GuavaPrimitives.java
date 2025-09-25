import java.util.AbstractList;
import java.util.Collections;
import java.util.List;
import org.checkerframework.checker.index.qual.HasSubsequence;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.MinLen;

public class GuavaPrimitives {

    @IndexOrLow("#1")
    @LessThan("#4")
    private static int indexOf(short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
        for (int i = start; i < end; i++) {
            if (array[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
